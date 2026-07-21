//! Design C read plane: a framed-JSON Unix-domain-socket transport that serves
//! ranked recall + ping from the single daemon-owned store.
//!
//! The read plane is **read-only by contract**: only ranked-recall and ping are
//! accepted, never a mutation, and the server forces
//! [`RecallOptions::record_access`](crate::RecallOptions) **off** so a read can
//! never bump usage counters. Frames are size-capped and `options.limit` is
//! clamped (hostile-input / OOM defence), and every connection is authenticated
//! by `SO_PEERCRED` UID before any request is served.
//!
//! Consumers use [`RankedRecallClient`] (`ipc` feature); the daemon serves via
//! [`IpcServer`] (`ipc` + `persistent`).

use std::io::{Read, Write};
#[cfg(all(feature = "ipc", feature = "persistent"))]
use std::os::unix::fs::FileTypeExt;
#[cfg(all(feature = "ipc", feature = "persistent"))]
use std::os::unix::io::AsRawFd;
use std::os::unix::net::UnixStream;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::coord::{io_err, CoordConfig};
use crate::errors::{MemoryError, Result};
use crate::{RecallOptions, Scored, SemanticFact};

/// Read/write timeout on the server socket so a wedged or idle peer cannot hang
/// a reader indefinitely (and so the server observes shutdown).
#[cfg(all(feature = "ipc", feature = "persistent"))]
const IO_TIMEOUT: Duration = Duration::from_millis(200);

/// Server-side hard cap on `RecallOptions.limit` (clamp, never honour verbatim).
#[cfg(all(feature = "ipc", feature = "persistent"))]
const MAX_RECALL_LIMIT: usize = 10_000;

// ---------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum IpcRequest {
    Ping,
    RecallFactsRanked {
        query: String,
        options: RecallOptions,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "ok", content = "value", rename_all = "snake_case")]
enum IpcResponse {
    Pong,
    RankedFacts(Vec<Scored<SemanticFact>>),
    Error(String),
}

/// Write a length-prefixed JSON frame in a single `write_all`.
fn write_frame<W: Write>(w: &mut W, payload: &[u8], max_frame: u32) -> Result<()> {
    let len = u32::try_from(payload.len()).map_err(|_| {
        MemoryError::Storage(format!("ipc frame too large: {} bytes", payload.len()))
    })?;
    if len > max_frame {
        return Err(MemoryError::Storage(format!(
            "ipc frame length {len} exceeds cap {max_frame}"
        )));
    }
    let mut framed = Vec::with_capacity(4 + payload.len());
    framed.extend_from_slice(&len.to_be_bytes());
    framed.extend_from_slice(payload);
    w.write_all(&framed)
        .map_err(|e| io_err("ipc-write-frame", e))?;
    w.flush().map_err(|e| io_err("ipc-flush", e))
}

/// Read a length-prefixed JSON frame, rejecting an oversized prefix *before*
/// allocating the body.
fn read_frame<R: Read>(r: &mut R, max_frame: u32) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)
        .map_err(|e| io_err("ipc-read-len", e))?;
    let len = u32::from_be_bytes(len_buf);
    if len > max_frame {
        return Err(MemoryError::Storage(format!(
            "ipc frame length {len} exceeds cap {max_frame}"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)
        .map_err(|e| io_err("ipc-read-body", e))?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// SO_PEERCRED peer authentication (Linux). Declared locally to avoid a `libc`
// dependency for two stable syscalls.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "ipc", feature = "persistent"))]
mod peercred {
    use super::*;

    #[repr(C)]
    struct Ucred {
        pid: i32,
        uid: u32,
        gid: u32,
    }

    const SOL_SOCKET: std::os::raw::c_int = 1;
    const SO_PEERCRED: std::os::raw::c_int = 17;

    extern "C" {
        fn getsockopt(
            fd: std::os::raw::c_int,
            level: std::os::raw::c_int,
            optname: std::os::raw::c_int,
            optval: *mut std::os::raw::c_void,
            optlen: *mut u32,
        ) -> std::os::raw::c_int;
        fn getuid() -> u32;
    }

    /// Reject a peer whose `SO_PEERCRED` UID does not match this process's UID.
    pub(super) fn check_same_uid(stream: &UnixStream) -> Result<()> {
        let mut cred = Ucred {
            pid: 0,
            uid: 0,
            gid: 0,
        };
        let mut len = std::mem::size_of::<Ucred>() as u32;
        // SAFETY: valid fd; `cred`/`len` are correctly-sized out-params for the
        // SO_PEERCRED getsockopt contract on Linux.
        let rc = unsafe {
            getsockopt(
                stream.as_raw_fd(),
                SOL_SOCKET,
                SO_PEERCRED,
                std::ptr::addr_of_mut!(cred).cast(),
                &mut len,
            )
        };
        if rc != 0 {
            return Err(io_err("ipc-peercred", std::io::Error::last_os_error()));
        }
        // SAFETY: `getuid` has no preconditions and cannot fail.
        let me = unsafe { getuid() };
        if cred.uid != me {
            return Err(MemoryError::SecurityViolation(
                "ipc peer uid mismatch".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Server (daemon: feature `ipc` + `persistent`)
// ---------------------------------------------------------------------------

/// Outcome of a framed read on a live server connection, distinguishing an idle
/// timeout (keep the connection) from a real close/error (drop it).
#[cfg(all(feature = "ipc", feature = "persistent"))]
enum FrameRead {
    Frame(Vec<u8>),
    Timeout,
    Closed,
    Error,
}

/// Read one length-prefixed frame from a connection, classifying the failure so
/// the server never busy-loops on a closed peer nor drops an idle one.
#[cfg(all(feature = "ipc", feature = "persistent"))]
fn read_frame_conn(stream: &mut UnixStream, max_frame: u32) -> FrameRead {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) => {
            return match e.kind() {
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => FrameRead::Timeout,
                std::io::ErrorKind::UnexpectedEof => FrameRead::Closed,
                _ => FrameRead::Error,
            };
        }
    }
    let len = u32::from_be_bytes(len_buf);
    if len > max_frame {
        return FrameRead::Error;
    }
    let mut buf = vec![0u8; len as usize];
    match stream.read_exact(&mut buf) {
        Ok(()) => FrameRead::Frame(buf),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => FrameRead::Closed,
        Err(_) => FrameRead::Error,
    }
}

/// The read server. Owns the listening socket under the coord dir and serves
/// ranked-recall/ping from a daemon-owned [`CognitiveMemory`](crate::CognitiveMemory).
#[cfg(all(feature = "ipc", feature = "persistent"))]
pub struct IpcServer {
    listener: std::os::unix::net::UnixListener,
    config: CoordConfig,
}

#[cfg(all(feature = "ipc", feature = "persistent"))]
impl IpcServer {
    /// Bind `read.sock` (`0o600`) under the coordination directory, provisioning
    /// the coord dir if necessary. Stale-socket removal is symlink-safe.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the directory or socket cannot be created
    /// with the required permissions, or a non-socket already occupies the path.
    pub fn bind(config: &CoordConfig) -> Result<Self> {
        use std::os::unix::fs::PermissionsExt;
        crate::coord::ensure_coord_dirs(config)?;
        let socket_path = config.socket_path();

        match std::fs::symlink_metadata(&socket_path) {
            Ok(meta) => {
                let ft = meta.file_type();
                if ft.is_symlink() {
                    return Err(MemoryError::SecurityViolation(format!(
                        "refusing to bind ipc socket: {} is a symlink",
                        socket_path.display()
                    )));
                }
                if ft.is_socket() {
                    std::fs::remove_file(&socket_path)
                        .map_err(|e| io_err("ipc-remove-stale-socket", e))?;
                } else {
                    return Err(MemoryError::Storage(format!(
                        "refusing to bind ipc socket: {} exists and is not a socket",
                        socket_path.display()
                    )));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_err("ipc-stat-socket", e)),
        }

        let listener = std::os::unix::net::UnixListener::bind(&socket_path)
            .map_err(|e| io_err("ipc-bind", e))?;
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| io_err("ipc-chmod-socket", e))?;

        Ok(Self {
            listener,
            config: config.clone(),
        })
    }

    /// Serve read requests against `memory` until `shutdown` returns `true`.
    /// Every connection is UID-checked via `SO_PEERCRED` before any request.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on an unrecoverable transport failure.
    pub fn serve(
        &self,
        memory: &mut crate::CognitiveMemory,
        shutdown: impl Fn() -> bool,
    ) -> Result<()> {
        self.accept_loop(shutdown, |query, options| {
            crate::CognitiveMemory::recall_facts_ranked(memory, query, options)
        })
    }

    /// Serve against a store shared behind a mutex (used by
    /// [`Coordinator::serve_reads`](crate::coord::Coordinator::serve_reads)),
    /// locking per request so the applier keeps making progress.
    pub(crate) fn serve_shared(
        &self,
        memory: &std::sync::Arc<std::sync::Mutex<crate::CognitiveMemory>>,
        shutdown: impl Fn() -> bool,
    ) -> Result<()> {
        self.accept_loop(shutdown, |query, options| {
            let mut mem = memory
                .lock()
                .map_err(|_| MemoryError::Storage("ipc shared store mutex poisoned".to_string()))?;
            mem.recall_facts_ranked(query, options)
        })
    }

    /// Single-active-connection accept loop. Uses a short socket timeout so it
    /// polls `shutdown` even while a client is idle.
    fn accept_loop(
        &self,
        shutdown: impl Fn() -> bool,
        mut recall: impl FnMut(&str, RecallOptions) -> Result<Vec<Scored<SemanticFact>>>,
    ) -> Result<()> {
        self.listener
            .set_nonblocking(true)
            .map_err(|e| io_err("ipc-set-nonblocking", e))?;

        while !shutdown() {
            match self.listener.accept() {
                Ok((stream, _addr)) => {
                    if peercred::check_same_uid(&stream).is_err() {
                        // Reject the peer and keep serving others.
                        continue;
                    }
                    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
                    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
                    self.serve_connection(stream, &shutdown, &mut recall);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(io_err("ipc-accept", e)),
            }
        }
        Ok(())
    }

    /// Serve requests on one connection until it closes or shutdown is signalled.
    fn serve_connection(
        &self,
        mut stream: UnixStream,
        shutdown: &impl Fn() -> bool,
        recall: &mut impl FnMut(&str, RecallOptions) -> Result<Vec<Scored<SemanticFact>>>,
    ) {
        let max_frame = self.config.max_frame_bytes;
        loop {
            if shutdown() {
                return;
            }
            let frame = match read_frame_conn(&mut stream, max_frame) {
                FrameRead::Frame(f) => f,
                // Idle client: no bytes within the read timeout. Re-check
                // shutdown and keep the connection open for the next request.
                FrameRead::Timeout => continue,
                // Peer closed, protocol/size violation, or fatal I/O: drop it.
                FrameRead::Closed | FrameRead::Error => return,
            };
            let response = match serde_json::from_slice::<IpcRequest>(&frame) {
                Ok(IpcRequest::Ping) => IpcResponse::Pong,
                Ok(IpcRequest::RecallFactsRanked { query, mut options }) => {
                    // Read-only by contract: never mutate the store on a read.
                    options.record_access = false;
                    options.limit = options.limit.min(MAX_RECALL_LIMIT);
                    match recall(&query, options) {
                        Ok(hits) => IpcResponse::RankedFacts(hits),
                        Err(e) => IpcResponse::Error(e.to_string()),
                    }
                }
                Err(e) => IpcResponse::Error(format!("malformed request: {e}")),
            };
            let bytes = match serde_json::to_vec(&response) {
                Ok(b) => b,
                Err(_) => return,
            };
            if write_frame(&mut stream, &bytes, max_frame).is_err() {
                return;
            }
        }
    }
}

#[cfg(all(feature = "ipc", feature = "persistent"))]
impl Drop for IpcServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.config.socket_path());
    }
}

// ---------------------------------------------------------------------------
// Client (consumer: feature `ipc`)
// ---------------------------------------------------------------------------

/// Connects to an [`IpcServer`] and issues read-only ranked-recall queries.
pub struct RankedRecallClient {
    stream: UnixStream,
    max_frame: u32,
}

impl RankedRecallClient {
    /// Connect to the read socket under the coordination directory.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] if the socket is absent or the connection is
    /// refused.
    pub fn connect(config: &CoordConfig) -> Result<Self> {
        let stream =
            UnixStream::connect(config.socket_path()).map_err(|e| io_err("ipc-connect", e))?;
        let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
        Ok(Self {
            stream,
            max_frame: config.max_frame_bytes,
        })
    }

    fn call(&self, req: &IpcRequest) -> Result<IpcResponse> {
        // UnixStream implements Read/Write for `&UnixStream`, so a shared borrow
        // suffices — the client API stays `&self`.
        let mut s = &self.stream;
        let bytes = serde_json::to_vec(req)
            .map_err(|e| MemoryError::Storage(format!("ipc serialize-request: {e}")))?;
        write_frame(&mut s, &bytes, self.max_frame)?;
        let resp = read_frame(&mut s, self.max_frame)?;
        serde_json::from_slice(&resp)
            .map_err(|e| MemoryError::Storage(format!("ipc parse-response: {e}")))
    }

    /// Liveness/health probe.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on any transport/protocol failure.
    pub fn ping(&self) -> Result<()> {
        match self.call(&IpcRequest::Ping)? {
            IpcResponse::Pong => Ok(()),
            IpcResponse::Error(msg) => Err(MemoryError::Storage(format!("ipc ping failed: {msg}"))),
            IpcResponse::RankedFacts(_) => Err(MemoryError::Storage(
                "ipc ping: unexpected ranked response".into(),
            )),
        }
    }

    /// Ranked recall over the daemon-owned store. `options.record_access` is
    /// forced off and `options.limit` clamped server-side; the result is the
    /// same `Vec<Scored<SemanticFact>>` an in-process recall returns.
    ///
    /// # Errors
    /// [`MemoryError::Storage`] on any transport/protocol/remote-scoring failure
    /// — there is **no silent fallback** to an unranked search.
    pub fn recall_facts_ranked(
        &self,
        query: &str,
        options: RecallOptions,
    ) -> Result<Vec<Scored<SemanticFact>>> {
        match self.call(&IpcRequest::RecallFactsRanked {
            query: query.to_string(),
            options,
        })? {
            IpcResponse::RankedFacts(hits) => Ok(hits),
            IpcResponse::Error(msg) => Err(MemoryError::Storage(msg)),
            IpcResponse::Pong => Err(MemoryError::Storage(
                "ipc recall: unexpected pong response".into(),
            )),
        }
    }
}
