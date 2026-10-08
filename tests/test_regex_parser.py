import itertools
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.automata import DFA, NFA, Regex, RegexSyntaxError, automaton_to_regex, equivalent, parse_regex, regex_to_nfa
from sofic.testing.strategies import dfas, nfas


def _words(alphabet, max_length):
    for length in range(max_length + 1):
        yield from itertools.product(alphabet, repeat=length)


def _assert_same_words(left, right, alphabet, max_length):
    for word in _words(sorted(alphabet, key=repr), max_length):
        assert left.recognizes(word) == right.recognizes(word), word


@pytest.mark.parametrize(
    "text",
    ["(a", "a)", "(a|b", "a|b)", "*a", "a|*", "+", "?a", "a|", "|a", "|", "a||b", "a**", "a+?", "", "()", "(|a)"],
)
def test_parse_errors_raise_regex_syntax_error(text):
    with pytest.raises(RegexSyntaxError) as info:
        parse_regex(text)
    assert isinstance(info.value, ValueError)
    assert 0 <= info.value.position <= len(text)


@pytest.mark.parametrize("text", ["'ab", "a\\", ".", "[ab]", "a{2}", "^a$", "(?=a)"])
def test_unsupported_or_unterminated_syntax_raises(text):
    with pytest.raises(RegexSyntaxError):
        parse_regex(text)


def test_parse_tree_is_unsimplified():
    assert parse_regex("ab|c*") == Regex(
        "union",
        (
            Regex("concat", (Regex("literal", ("a",)), Regex("literal", ("b",)))),
            Regex("star", (Regex("literal", ("c",)),)),
        ),
    )
    assert parse_regex("'ab'") == Regex("literal", ("ab",))
    assert parse_regex("ε") == parse_regex("(?:)") == Regex("epsilon")
    assert parse_regex("∅") == parse_regex("(?!)") == Regex("empty")


@pytest.mark.parametrize(
    ("text", "python"),
    [
        ("a", "a"),
        ("ab|ba", "ab|ba"),
        ("(ab)*", "(ab)*"),
        ("(?:a|b)*abb", "(?:a|b)*abb"),
        ("a+b?", "a+b?"),
        ("(a|ε)b", "(a|)b"),
        ("ε", ""),
        ("∅", "(?!)"),
        ("a∅|b", "a(?!)|b"),
        ("(a*b*)*c", "(a*b*)*c"),
        ("((a|b)(a|b))*", "((a|b)(a|b))*"),
        ("a(?:)b", "ab"),
    ],
)
def test_known_languages_against_python_re(text, python):
    nfa = regex_to_nfa(text, alphabet={"a", "b", "c"})
    compiled = re.compile(python)
    for word in _words("abc", 6):
        assert nfa.recognizes(word) == bool(compiled.fullmatch("".join(word))), (text, word)


def test_multi_character_and_escaped_symbols():
    nfa = regex_to_nfa("'ab'(a|b)*\\*'it\\'s'")
    assert nfa.input_alphabet == frozenset({"ab", "a", "b", "*", "it's"})
    assert nfa.recognizes(("ab", "*", "it's"))
    assert nfa.recognizes(("ab", "a", "b", "*", "it's"))
    assert not nfa.recognizes(("a", "b", "*", "it's"))


def test_thompson_shape():
    nfa = regex_to_nfa("(a|b)*c")
    assert len(nfa.initial_states) == 1
    assert len(nfa.accepting_states) == 1
    assert len(list(nfa.states())) <= 2 * 6


def test_alphabet_resolves_non_string_symbols():
    nfa = regex_to_nfa("0(1|'10')*", alphabet={0, 1, 10})
    assert nfa.input_alphabet == frozenset({0, 1, 10})
    assert nfa.recognizes((0, 10, 1))
    with pytest.raises(ValueError, match="not in"):
        regex_to_nfa("2", alphabet={0, 1})


def test_nfa_from_regex_delegates():
    assert equivalent(NFA.from_regex("(ab)*"), regex_to_nfa("(ab)*"))


def test_automaton_to_regex_quotes_multi_character_symbols():
    def single(symbol):
        dfa = DFA(input_alphabet=frozenset({"a", "b", "ab"}), initial_states=frozenset({0}))
        dfa.graph.add_state(0)
        dfa.graph.add_state(1)
        dfa.accepting_states = frozenset({1})
        if symbol == "a b":
            dfa.graph.add_state(2)
            dfa.add_transition(0, 2, "a")
            dfa.add_transition(2, 1, "b")
        else:
            dfa.add_transition(0, 1, symbol)
        return dfa

    assert automaton_to_regex(single("ab")) == "'ab'"
    assert automaton_to_regex(single("a b")) == "ab"


def test_automaton_to_regex_single_character_output_is_python_re():
    dfa = DFA(input_alphabet=frozenset({"a", "b"}), initial_states=frozenset({0}), accepting_states=frozenset({0}))
    dfa.graph.add_state(0)
    dfa.graph.add_state(1)
    dfa.add_transition(0, 1, "a")
    dfa.add_transition(1, 0, "b")
    pattern = automaton_to_regex(dfa)
    assert pattern == "(?:(?:)|a(?:ba)*b)"
    for word in _words("ab", 6):
        assert bool(re.fullmatch(pattern, "".join(word))) == dfa.recognizes(word)


_SYMBOL_ALPHABETS = [
    ("0", "1"),
    ("a", "ab", "b"),
    ("ε", "∅", "'"),
    ("(", "|", "x y", "\\", "a'b"),
]


@settings(max_examples=60, deadline=None)
@given(st.sampled_from(_SYMBOL_ALPHABETS).flatmap(lambda alphabet: nfas(alphabet=alphabet, max_states=3)))
def test_round_trip_nfa(nfa):
    rebuilt = regex_to_nfa(automaton_to_regex(nfa), alphabet=nfa.input_alphabet)
    _assert_same_words(nfa, rebuilt, nfa.input_alphabet, 4)
    assert equivalent(nfa, rebuilt, nfa.input_alphabet)


@settings(max_examples=60, deadline=None)
@given(st.sampled_from(_SYMBOL_ALPHABETS[:3]).flatmap(lambda alphabet: dfas(alphabet=alphabet, max_states=3)))
def test_round_trip_dfa(dfa):
    rebuilt = regex_to_nfa(automaton_to_regex(dfa), alphabet=dfa.input_alphabet)
    _assert_same_words(dfa, rebuilt, dfa.input_alphabet, 4)
    assert equivalent(dfa, rebuilt, dfa.input_alphabet)


@settings(max_examples=40, deadline=None)
@given(dfas(alphabet=(0, 1, 10), max_states=3))
def test_round_trip_non_string_symbols(dfa):
    rebuilt = regex_to_nfa(automaton_to_regex(dfa), alphabet=dfa.input_alphabet)
    assert equivalent(dfa, rebuilt, dfa.input_alphabet)
