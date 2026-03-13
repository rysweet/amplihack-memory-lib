//! Shared utility functions.

/// Hex-encode a byte slice to a lowercase hex string.
pub(crate) fn hex_encode(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}
