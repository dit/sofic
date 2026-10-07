"""Tests for regular-language wrappers and quotients."""

from sofic.automata.dfa import DFA
from sofic.automata.languages.base import AutomatonLanguage, ExplicitLanguage, as_language
from sofic.automata.languages.quotients import left_quotient, right_quotient
from sofic.automata.languages.residuals import is_composed_residual


def _simple_dfa() -> DFA:
    dfa = DFA(
        input_alphabet=frozenset({"a", "b"}), initial_states=frozenset({"q0"}), accepting_states=frozenset({"q1"})
    )
    dfa.graph.add_state("q0")
    dfa.graph.add_state("q1")
    dfa.add_transition("q0", "q1", "a")
    dfa.add_transition("q0", "q0", "b")
    dfa.add_transition("q1", "q1", "a")
    dfa.add_transition("q1", "q0", "b")
    return dfa


def test_explicit_language_membership():
    lang = ExplicitLanguage(
        positive={("a",), ("ab",)},
        negative={("b",)},
        alphabet=frozenset({"a", "b"}),
    )
    assert ("a",) in lang
    assert ("ab",) in lang
    assert ("b",) not in lang
    assert ("aa",) not in lang


def test_automaton_language():
    dfa = _simple_dfa()
    lang = AutomatonLanguage(dfa)
    assert ("a",) in lang
    assert ("b",) not in lang
    assert as_language(dfa) is not dfa
    assert ("a",) in as_language(dfa)


def test_left_quotient():
    lang = ExplicitLanguage(positive={("a", "b"), ("a", "c")}, alphabet=frozenset({"a", "b", "c"}))
    quot = left_quotient(("a",), lang)
    assert isinstance(quot, ExplicitLanguage)
    assert ("b",) in quot
    assert ("c",) in quot
    assert ("a",) not in quot


def test_right_quotient():
    lang = ExplicitLanguage(positive={("b", "a"), ("c", "a")}, alphabet=frozenset({"a", "b", "c"}))
    quot = right_quotient(lang, ("a",))
    assert ("b",) in quot
    assert ("c",) in quot


def test_is_composed_residual():
    r1 = ExplicitLanguage(positive={("a",)})
    r2 = ExplicitLanguage(positive={("b",)})
    union = ExplicitLanguage(positive={("a",), ("b",)})
    all_residuals = frozenset({r1, r2, union})
    assert is_composed_residual(union, all_residuals)
    assert not is_composed_residual(r1, all_residuals)


def test_atoms_from_dfa():
    from sofic.automata.languages.atoms import atoms

    dfa = _simple_dfa()
    result = atoms(AutomatonLanguage(dfa))
    assert len(result) >= 1


def _brute_words(alphabet, length):
    import itertools

    return [w for n in range(length + 1) for w in itertools.product(sorted(alphabet), repeat=n)]


def _random_dfa(rng, n):
    dfa = DFA(
        input_alphabet=frozenset("ab"),
        initial_states=frozenset({0}),
        accepting_states=frozenset(s for s in range(n) if rng.random() < 0.4),
    )
    for state in range(n):
        dfa.graph.add_state(state)
    for state in range(n):
        for symbol in "ab":
            if rng.random() < 0.75:
                dfa.add_transition(state, rng.randrange(n), symbol)
    return dfa


def _empty_dfa() -> DFA:
    dfa = DFA(input_alphabet=frozenset("ab"), initial_states=frozenset({0}), accepting_states=frozenset())
    dfa.graph.add_state(0)
    dfa.add_transition(0, 0, "a")
    return dfa


def test_boolean_operations_with_an_empty_operand():
    empty, other = _empty_dfa(), _simple_dfa()
    words = _brute_words("ab", 4)
    assert not any(empty.intersection(other).recognizes(w) for w in words)
    assert not any(other.intersection(empty).recognizes(w) for w in words)
    assert not any(empty.difference(other).recognizes(w) for w in words)
    assert all(other.difference(empty).recognizes(w) == other.recognizes(w) for w in words)
    assert all(empty.complement().recognizes(w) for w in words)


def test_minimizers_agree_on_trimmed_form():
    import random

    rng = random.Random(4)
    for _trial in range(200):
        dfa = _random_dfa(rng, rng.randint(1, 4))
        results = [dfa.minimize(algorithm=name) for name in ("hopcroft", "moore", "brzozowski")]
        sizes = {len(list(result.states())) for result in results}
        assert len(sizes) == 1
        words = _brute_words("ab", 5)
        for result in results:
            assert all(result.recognizes(w) == dfa.recognizes(w) for w in words)


def test_package_exports_the_residuals_function():
    from sofic.automata import languages

    assert callable(languages.residuals)
    assert len(languages.residuals(AutomatonLanguage(_simple_dfa()))) == 2


def test_left_quotients_include_the_empty_residual():
    import random

    from sofic.automata.languages.quotients import left_quotients

    rng = random.Random(5)
    words = _brute_words("ab", 4)
    prefixes = _brute_words("ab", 4)
    for _trial in range(100):
        dfa = _random_dfa(rng, rng.randint(1, 4))
        expected = {frozenset(w for w in words if dfa.recognizes(u + w)) for u in prefixes}
        got = {frozenset(w for w in words if w in q) for q in left_quotients(AutomatonLanguage(dfa))}
        assert got == expected


def test_atoms_partition_sigma_star():
    import random

    from sofic.automata.languages.atoms import atoms

    rng = random.Random(6)
    words = _brute_words("ab", 5)
    prefixes = _brute_words("ab", 4)
    for _trial in range(100):
        dfa = _random_dfa(rng, rng.randint(1, 4))
        found = atoms(AutomatonLanguage(dfa))
        for w in words:
            assert sum(w in atom for atom in found) == 1
        # Each atom is a class of words with equal quotient membership pattern.
        signatures = {tuple(dfa.recognizes(u + w) for u in prefixes) for w in words}
        assert len(found) >= len({s for s in signatures})


def test_empty_language_has_one_atom_and_one_residual():
    from sofic.automata.languages.atoms import atoms
    from sofic.automata.languages.quotients import left_quotients

    language = AutomatonLanguage(_empty_dfa())
    assert len(left_quotients(language)) == 1
    [atom] = atoms(language)
    assert all(w in atom for w in _brute_words("ab", 3))


def test_right_quotient_by_the_empty_word_is_identity():
    lang = ExplicitLanguage(positive={("a",), ("a", "b")}, negative={("b",)})
    quotient = right_quotient(lang, ())
    assert quotient._positive == {("a",), ("a", "b")}
    assert quotient._negative == {("b",)}


def test_composed_residual_ignores_non_contained_residuals():
    r1 = ExplicitLanguage(positive={("a",)})
    r2 = ExplicitLanguage(positive={("a",), ("b",)})
    bigger = ExplicitLanguage(positive={("a",), ("b",), ("c",)})
    assert not is_composed_residual(r2, frozenset({r1, r2, bigger}))
