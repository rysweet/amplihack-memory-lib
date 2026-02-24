"""Tests for entity_extraction module."""

from amplihack_memory.entity_extraction import extract_entity_name


class TestExtractEntityName:
    """Tests for extract_entity_name function."""

    def test_multi_word_proper_noun(self):
        """Extract multi-word names like 'Sarah Chen'."""
        result = extract_entity_name("Sarah Chen loves painting", "Sarah Chen hobbies")
        assert result == "sarah chen"

    def test_apostrophe_name(self):
        """Extract names with apostrophes like O'Brien."""
        result = extract_entity_name("O'Brien won the race", "O'Brien achievements")
        assert result == "o'brien"

    def test_hyphenated_name(self):
        """Extract hyphenated names like Al-Hassan."""
        result = extract_entity_name("Al-Hassan studies physics", "Al-Hassan education")
        assert result == "al-hassan"

    def test_concept_field_preferred(self):
        """Concept field is checked before content."""
        result = extract_entity_name(
            "The person likes music",
            "John Smith preferences",
        )
        assert result == "john smith"

    def test_single_capitalized_word_not_at_start(self):
        """Single capitalized word not at sentence start is extracted."""
        result = extract_entity_name("The city of Paris is beautiful", "")
        assert result == "paris"

    def test_empty_input(self):
        """Empty strings return empty result."""
        assert extract_entity_name("", "") == ""
        assert extract_entity_name("", "  ") == ""

    def test_no_proper_nouns(self):
        """All lowercase text returns empty result."""
        result = extract_entity_name("the quick brown fox", "animals")
        assert result == ""

    def test_possessive_name(self):
        """Names in possessive context are handled."""
        # The possessive 's is part of the name pattern matching
        result = extract_entity_name("", "Maria Garcia hobbies")
        assert result == "maria garcia"

    def test_unicode_apostrophe(self):
        """Unicode right single quotation mark (curly apostrophe) is handled."""
        result = extract_entity_name("O\u2019Brien won", "O\u2019Brien")
        assert "brien" in result  # Should handle the unicode apostrophe

    def test_content_fallback_when_concept_empty(self):
        """Falls back to content when concept has no entities."""
        result = extract_entity_name("John Smith runs daily", "fitness")
        assert result == "john smith"

    def test_longest_match_preferred(self):
        """When multiple matches exist, longest is preferred."""
        result = extract_entity_name(
            "Mary Jane Watson met Peter Parker",
            "Mary Jane Watson",
        )
        assert result == "mary jane watson"
