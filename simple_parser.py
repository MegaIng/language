from __future__ import annotations

from abc import ABCMeta, abstractmethod
from reprlib import recursive_repr
from threading import get_ident
from typing import Iterator, TypeVar, Iterable, Generic, List, Callable, overload, Optional, Dict, Container, \
    Type, Any, Tuple


def recursion_protected():
    def decorating_function(user_function):
        current = set()

        def wrapper(self, *args, **kwargs):
            key = id(self), get_ident()
            if key in current:
                return
            current.add(key)
            try:
                result = user_function(self, *args, **kwargs)
            finally:
                current.discard(key)
            return result

        return wrapper

    return decorating_function


T = TypeVar('T')


class Rewindable(Iterator[T]):
    """
    A Rewindable Wrapper around an iterator the allows for 'rewinding'
    the iterator by reloading an old state
    """

    def __init__(self, iterator: Iterable[T]):
        self._iterator = iter(iterator)
        self._index = 0
        self._cache = []

    def __next__(self) -> T:
        while self._index + 1 > len(self._cache):
            self._cache.append(next(self._iterator))
        temp = self._cache[self._index]
        self._index += 1
        return temp

    def get_state(self) -> object:
        """Returns an object indicating the current position
        that can be reloaded via `load_state`"""
        return self._index

    def load_state(self, state: object):
        """Reloads a state returned by `get_state`"""
        assert isinstance(state, int)
        self._index = state

    def peek(self) -> T:
        while self._index + 1 > len(self._cache):
            self._cache.append(next(self._iterator))
        return self._cache[self._index]

    def has_next(self) -> bool:
        try:
            self.peek()
        except StopIteration:
            return False
        return True


IN = TypeVar('IN')
OUT = TypeVar('OUT')
INNER_OUT = TypeVar('INNER_OUT')
TRANSFORM_FUNCTION = Callable[[List[INNER_OUT]], Iterable[OUT]]


class Parser(Generic[IN, OUT], metaclass=ABCMeta):
    """
    Base class for Parser which match pattern in an rewindable input stream and return all possible matches
    """

    def __init__(self, name: str = None):
        self._name = name

    def set_name(self, name: str):
        self._name = name

    def register(self, context: Dict[str, Parser]):
        if self._name is not None:
            context[self._name] = self
        return self

    @abstractmethod
    def parse(self, in_stream: Rewindable[IN]) -> Iterator[Iterator[OUT]]:
        """
        Matches the pattern in the Rewindable stream `in_stream`
        Returns an Iterator over all possible matches (also iterator)
        The implementation must insure that `in_stream` is reset after StopIteration is thrown
        """
        raise NotImplementedError

    def resolve_references(self, context: Dict[str, Parser]):
        """
        All ReferenceParser should be resolved to the referenced Parser
        """
        pass


class Match(Parser[IN, OUT]):
    """
    A Parser that matches the value `expected` and yields `result` in response
    """

    def __init__(self, expected: IN, result: OUT = None, name: str = None):
        super().__init__(name)
        self.expected = expected
        self.result = result

    def __repr__(self):
        return f"<{self.expected!r}>"

    def parse(self, in_stream: Rewindable[IN]) -> Iterator[Iterator[OUT]]:
        state = in_stream.get_state()
        try:
            n = next(in_stream)
        except StopIteration:
            in_stream.load_state(state)
            return
        if n == self.expected:
            yield iter((self.result,))
        in_stream.load_state(state)


class ExtendedMatch(Parser[IN, OUT]):
    def __init__(self, class_: Type[IN] = None, attributes: List[Tuple[str, Any]] = None,
                 transform_function: Callable[[IN], Iterable[OUT]] = lambda x: (x,), name: str = None):
        super(ExtendedMatch, self).__init__(name)
        self._transform_function = transform_function
        self._class = class_
        self._attributes = attributes

    def __repr__(self):
        if self._attributes is None:
            return f"<{self._class.__name__}>"
        else:
            return f"<{self._class.__name__}: {' '.join(f'{n}={v!r}' for n,v in self._attributes)}>"

    def parse(self, in_stream: Rewindable[IN]):
        state = in_stream.get_state()
        try:
            n = next(in_stream)
        except StopIteration:
            in_stream.load_state(state)
            return
        success = True
        if self._class is not None and not isinstance(n, self._class):
            success = False
        if self._attributes is not None:
            for name, v in self._attributes:
                if getattr(n, name) != v:
                    success = False
                    break
        if success:
            r = self._transform_function(n)
            yield iter(r)
        in_stream.load_state(state)


class Transform(Parser[IN, OUT]):
    def __init__(self, parser: Parser[IN, INNER_OUT], transform_function: TRANSFORM_FUNCTION):
        super(Transform, self).__init__(None)
        self._transform_function = transform_function
        self._parser = parser

    @recursive_repr()
    def __repr__(self):
        return repr(self._parser)

    def parse(self, in_stream: Rewindable[IN]):
        for m in self._parser.parse(in_stream):
            yield self._transform_function(list(m))

    @recursion_protected()
    def resolve_references(self, context: Dict[str, Parser]):
        if isinstance(self._parser, Reference):
            self._parser = context[self._parser._target_name]
        self._parser.resolve_references(context)


class AnyOf(Parser[IN, OUT]):
    def __init__(self, options: Container[IN], transform_function: Callable[[IN], OUT] = lambda x: (x,),
                 name: str = None):
        super().__init__(name)
        self._options = options
        self._tf = transform_function

    def __repr__(self):
        if isinstance(self._options, Iterable):
            return "(" + '|'.join(f"<{o!r}>" for o in self._options)
        else:
            return f"({self._options!r})"

    def parse(self, in_stream: Rewindable[IN]):
        state = in_stream.get_state()
        try:
            symbol = next(in_stream)
        except StopIteration:
            in_stream.load_state(state)
            return
        else:
            if symbol in self._options:
                yield iter(self._tf(symbol))
                in_stream.load_state(state)
            else:
                in_stream.load_state(state)
                return


class NOT(Parser[IN, OUT]):
    def __init__(self, parser: Parser[IN, INNER_OUT], name: str = None):
        super().__init__(name)
        self._parser = parser

    @recursive_repr()
    def __repr__(self):
        return f"(!{self._parser!r})"

    def parse(self, in_stream: Rewindable[IN]):
        it = self._parser.parse(in_stream)
        try:
            next(it)
        except StopIteration:
            yield iter(())
        else:
            for _ in it:
                pass
            return

    @recursion_protected()
    def resolve_references(self, context: Dict[str, Parser]):
        if isinstance(self._parser, Reference):
            self._parser = context[self._parser._target_name]
        self._parser.resolve_references(context)


class Discard(Parser[IN, OUT]):
    def __init__(self, parser: Parser[IN, INNER_OUT], name: str = None):
        super().__init__(name)
        self._parser = parser

    @recursive_repr()
    def __repr__(self):
        return repr(self._parser)

    def parse(self, in_stream: Rewindable[IN]):
        it = self._parser.parse(in_stream)
        for _ in it:
            yield iter(())


class AnyParser(Parser[IN, OUT]):
    def __init__(self, transform_function: Callable[[IN], OUT] = lambda x: (x,), name: str = None):
        super().__init__(name)
        self._transform_function = transform_function

    def __repr__(self):
        return "."

    def parse(self, in_stream: Rewindable[IN]):
        state = in_stream.get_state()
        try:
            yield iter(self._transform_function(next(in_stream)))
            in_stream.load_state(state)
        except StopIteration:
            in_stream.load_state(state)
            return


class OR(Parser[IN, OUT]):
    """
    A Parser that matches any Parser of `sub_parser` and returns all possible results of each one
    """

    def __init__(self, sub_parser: List[Parser[IN, OUT]], name: str = None):
        super().__init__(name)
        self._sub_parser = list(sub_parser)

    @recursive_repr()
    def __repr__(self):
        return '(' + '|'.join(repr(sp) for sp in self._sub_parser) + ')'

    def add_option(self, parser: Parser[IN, OUT]):
        self._sub_parser.append(parser)

    def parse(self, in_stream: Rewindable[IN]) -> Iterator[Iterator[OUT]]:
        for sp in self._sub_parser:
            yield from sp.parse(in_stream)

    @recursion_protected()
    def resolve_references(self, context: Dict[str, Parser]):
        for i, sp in enumerate(self._sub_parser):
            if isinstance(sp, Reference):
                self._sub_parser[i] = context[sp._target_name]
            self._sub_parser[i].resolve_references(context)


class AND(Parser[IN, OUT]):
    """
    A  Parser that matches the Parsers of `sub_parser` and returns all possible combinations
    """
    _sub_parser: List
    _transform_function: TRANSFORM_FUNCTION

    @overload
    def __init__(self, sub_parser: List[Parser[IN, OUT]], name: str = None):
        raise NotImplementedError

    @overload
    def __init__(self, sub_parser: List[Parser[IN, INNER_OUT]],
                 transform_function: TRANSFORM_FUNCTION, name: str = None):
        raise NotImplementedError

    def __init__(self, sub_parser, transform_function=None, name: str = None):
        super().__init__(name)
        self._sub_parser = list(sub_parser)
        self._transform_function = transform_function

    @recursive_repr()
    def __repr__(self):
        return '(' + ' '.join(repr(sp) for sp in self._sub_parser) + ')'

    def parse(self, in_stream: Rewindable[IN]) -> Iterator[Iterator[OUT]]:
        iterator_stack = [self._sub_parser[0].parse(in_stream)]
        try:
            t = next(iterator_stack[0])
        except StopIteration:
            return
        result = [tuple(t)]
        skip = False  # should a new Iterator be appended
        while len(iterator_stack) > 0:
            if not skip:
                if len(iterator_stack) == len(self._sub_parser):
                    assert len(result) == len(self._sub_parser)
                    if self._transform_function is None:
                        yield (v for t in tuple(result) for v in t)
                    else:
                        yield iter(self._transform_function([v for t in tuple(result) for v in t]))
                else:
                    iterator_stack.append(self._sub_parser[len(iterator_stack)].parse(in_stream))
                    result.append(())
            try:
                result[-1] = tuple(next(iterator_stack[-1]))
                skip = False
            except StopIteration:
                skip = True
                result.pop()
                iterator_stack.pop()

    @recursion_protected()
    def resolve_references(self, context: Dict[str, Parser]):
        for i, sp in enumerate(self._sub_parser):
            if isinstance(sp, Reference):
                self._sub_parser[i] = context[sp._target_name]
            self._sub_parser[i].resolve_references(context)


class RepeatingNG(Parser[IN, OUT]):
    """
    A Parser that matches `min_count` to `max_count` repetitions of Parser `parser`
    and returns all possible combinations (Non Greedy, matches as few repetitions as possible first
    `max_count` can be `None` to indicate that the repetition can be infinite
    """
    _min_count: int
    _max_count: Optional[int]
    _parser: Parser
    _transform_function: TRANSFORM_FUNCTION

    @overload
    def __init__(self, parser: Parser[IN, OUT], min_count: int, max_count: int = None, name=None):
        ...

    @overload
    def __init__(self, parser: Parser[IN, INNER_OUT], min_count: int, max_count: int = None,
                 transform_function: TRANSFORM_FUNCTION = None, name=None):
        ...

    def __init__(self, parser: Parser[IN, OUT], min_count: int, max_count: int = None,
                 transform_function: TRANSFORM_FUNCTION = None, name: str = None):
        super().__init__(name)
        self._parser = parser
        self._min_count = min_count
        self._max_count = max_count
        self._transform_function = transform_function

    @recursive_repr()
    def __repr__(self):
        if self._max_count is None:
            return f"{self._parser!r}{{{self._min_count}:}}"
        elif self._max_count == self._min_count:
            return f"{self._parser!r}{{{self._min_count}}}"
        else:
            return f"{self._parser!r}{{{self._min_count}:{self._max_count}}}"

    def parse(self, in_stream: Rewindable[IN]) -> Iterator[Iterator[OUT]]:
        count = self._min_count
        if count == 0:  # Special case Optional matching
            yield iter(())
            count = 1
        while self._max_count is None or self._max_count > count:
            iterator_stack = [self._parser.parse(in_stream)]
            try:
                result = [tuple(next(iterator_stack[0]))]
            except StopIteration:
                return
            skip = False
            any_result = False
            while len(iterator_stack) > 0:
                if not skip:
                    if len(iterator_stack) >= count:
                        assert len(result) == len(iterator_stack)
                        any_result = True
                        if self._transform_function is None:
                            yield (v for t in result for v in t)
                        else:
                            yield iter(self._transform_function([v for t in result for v in t]))
                    else:
                        iterator_stack.append(self._parser.parse(in_stream))
                        result.append(())
                try:
                    result[-1] = tuple(next(iterator_stack[-1]))
                    skip = False
                except StopIteration:
                    skip = True
                    result.pop()
                    iterator_stack.pop()
            count += 1
            if not any_result:
                return

    @recursion_protected()
    def resolve_references(self, context: Dict[str, Parser]):
        if isinstance(self._parser, Reference):
            self._parser = context[self._parser._target_name]
        self._parser.resolve_references(context)


class OptionalNG(Parser[IN, OUT]):
    def __init__(self, parser: Parser[IN, INNER_OUT], name: str = None):
        super().__init__(name)
        self._parser = parser

    def parse(self, in_stream: Rewindable[IN]) -> Iterator[Iterator[OUT]]:
        yield iter(())
        yield from self._parser.parse(in_stream)

    @recursion_protected()
    def resolve_references(self, context: Dict[str, Parser]):
        if isinstance(self._parser, Reference):
            self._parser = context[self._parser._target_name]
        self._parser.resolve_references(context)


class Maybe(Parser[IN, OUT]):
    def __init__(self, parser: Parser[IN, INNER_OUT], name: str = None):
        super().__init__(name)
        self._parser = parser

    def parse(self, in_stream: Rewindable[IN]) -> Iterator[Iterator[OUT]]:
        it = self._parser.parse(in_stream)
        try:
            r = next(it)
        except StopIteration:
            yield iter(())
        else:
            yield r
            yield from it

    @recursion_protected()
    def resolve_references(self, context: Dict[str, Parser]):
        if isinstance(self._parser, Reference):
            self._parser = context[self._parser._target_name]
        self._parser.resolve_references(context)


class Reversed(Parser):
    def __init__(self, parser: Parser[IN, INNER_OUT], name: str = None):
        super().__init__(name)
        self._parser = parser

    def parse(self, in_stream: Rewindable[IN]) -> Iterator[Iterator[OUT]]:
        state = in_stream.get_state()
        results = []
        for r in self._parser.parse(in_stream):
            results.append((list(r), in_stream.get_state()))
        assert in_stream.get_state() == state
        for r, s in reversed(results):
            in_stream.load_state(s)
            yield iter(r)
        in_stream.load_state(state)

    @recursion_protected()
    def resolve_references(self, context: Dict[str, Parser]):
        if isinstance(self._parser, Reference):
            self._parser = context[self._parser._target_name]
        self._parser.resolve_references(context)


class EOI(Parser):
    """EOI - End of Input"""

    def parse(self, in_stream: Rewindable[IN]):
        try:
            in_stream.peek()
        except StopIteration:
            yield (())


class Reference(Parser):
    def __init__(self, target_name: str):
        super().__init__(None)
        self._target_name = target_name

    def __repr__(self):
        return self._target_name

    def parse(self, in_stream: Rewindable[IN]):
        raise NotImplementedError("resolve_references not called")

    @recursion_protected()
    def resolve_references(self, context: Dict[str, Parser]):
        ref = context[self._target_name]
        if isinstance(ref, Reference):
            self._target_name = ref._target_name
