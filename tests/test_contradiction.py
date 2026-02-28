"""Tests for contradiction detection module."""

from amplihack_memory.contradiction import detect_contradiction


class TestDetectContradiction:
    """Tests for detect_contradiction function."""

    def test_contradictory_numbers(self):
        """Two facts about the same concept with different numbers."""
        result = detect_contradiction(
            "Klaebo has won 9 gold medals",
            "Klaebo has won 10 gold medals",
            "Klaebo medals",
            "Klaebo medals",
        )
        assert result.get("contradiction") is True
        assert "conflicting_values" in result

    def test_no_contradiction_same_numbers(self):
        """Same numbers should not flag contradiction."""
        result = detect_contradiction(
            "The team scored 5 goals",
            "The team scored 5 goals in the match",
            "team goals",
            "team goals",
        )
        assert result == {}

    def test_no_contradiction_different_concepts(self):
        """Facts about different concepts should not flag contradiction."""
        result = detect_contradiction(
            "Temperature is 25 degrees",
            "Weight is 150 pounds",
            "temperature",
            "weight",
        )
        assert result == {}

    def test_empty_concepts(self):
        """Empty concepts return no contradiction."""
        assert detect_contradiction("fact 1", "fact 2", "", "") == {}
        assert detect_contradiction("fact 1", "fact 2", "", "concept") == {}
        assert detect_contradiction("fact 1", "fact 2", "concept", "") == {}

    def test_no_numbers(self):
        """Facts without numbers return no contradiction."""
        result = detect_contradiction(
            "Klaebo is a great skier",
            "Klaebo excels at skiing",
            "Klaebo skiing",
            "Klaebo skiing",
        )
        assert result == {}

    def test_short_concept_words_ignored(self):
        """Very short concept words (<=2 chars) are not considered for overlap."""
        result = detect_contradiction(
            "Team A scored 3 goals",
            "Team B scored 5 goals",
            "Team A",
            "Team B",
        )
        # "A" and "B" are <= 2 chars, only "team" matches
        assert result.get("contradiction") is True

    def test_decimal_numbers(self):
        """Decimal numbers are detected correctly."""
        result = detect_contradiction(
            "The price is 3.50 dollars",
            "The price is 4.75 dollars",
            "price item",
            "price item",
        )
        assert result.get("contradiction") is True

    def test_concept_word_overlap_required(self):
        """At least one meaningful concept word must overlap."""
        result = detect_contradiction(
            "Score was 10",
            "Score was 20",
            "xyz",
            "abc",
        )
        # Concepts share no words longer than 2 chars
        assert result == {}
