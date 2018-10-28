from dataclasses import dataclass
from decimal import Decimal
from pprint import pprint
from typing import Any, DefaultDict, Callable, Tuple, List

from simple_parser import AND, RepeatingNG, AnyOf, NOT, Discard, Match, Rewindable, OR, Reversed, OptionalNG, AnyParser, \
    Transform, ExtendedMatch, Reference, Maybe, EOI


@dataclass
class Token:
    type: str
    content: Any


def match_token(type_: str, content: str = None, tf=lambda x: (x,)):
    if content is None:
        return ExtendedMatch(Token, [('type', type_)], tf)
    else:
        return ExtendedMatch(Token, [('type', type_), ('content', content)], tf)


name_token = AND([
    AnyOf("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"),
    RepeatingNG(AnyOf("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"), 0),
    NOT(AnyOf("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789"))
], name='name_token', transform_function=lambda x: (Token('name', ''.join(x)),))

number_token = AND([
    RepeatingNG(AnyOf("0123456789"), 1),
    OptionalNG(AND([
        Match(".", '.'),
        RepeatingNG(AnyOf("0123456789"), 1),
    ])),
    OptionalNG(AND([
        Match("e",'e'),
        RepeatingNG(AnyOf("0123456789"), 1),
    ])),
], name='number_token', transform_function=lambda x: (Token('number', Decimal(''.join(x))),))

string_token = AND([
    Discard(Match('"')),
    Reversed(RepeatingNG(OR([
        AND([
            NOT(OR([Match("\\"), Match('"')])),
            AnyParser()
        ]),
        AND([
            Discard(Match("\\")),
            AnyParser()
        ])
    ]), 0)),
    Discard(Match('"')),
], name='string_token', transform_function=lambda x: (Token('string', ''.join(x)),))

space_token = RepeatingNG(Match(' '), 1)
newline_token = Match('\n', Token('newline', '\n'))

symbol_token = Transform(OR([
    Match("(", '('),
    Match(")", ')'),
    Match("[", '['),
    Match("]", ']'),
    Match("{", '{'),
    Match("}", '}'),
    AND([Match("-", '-'), Match(">", '>')]),
    Match(":", ':'),
]), transform_function=lambda x: (Token("symbol", ''.join(x)),))

lexer = Reversed(RepeatingNG(OR([
    name_token,
    Discard(space_token),
    newline_token,
    string_token,
    number_token,
    symbol_token
]), 1))

name = ExtendedMatch(Token, [('type', 'name')])
number = ExtendedMatch(Token, [('type', 'number')])
newline = ExtendedMatch(Token, [('type', 'newline')])
string = ExtendedMatch(Token, [('type', 'string')])

open_parentheses = match_token('symbol', '(')
close_parentheses = match_token('symbol', ')')
open_brace = match_token('symbol', '{')
close_brace = match_token('symbol', '}')
open_bracket = match_token('symbol', '[')
close_bracket = match_token('symbol', ']')
arrow = match_token('symbol', '->')
colon = match_token('symbol', ':')

typed_parameter = AND([
    Discard(open_brace),
    name,
    Discard(colon),
    name,
    Discard(close_brace),
], name='typed_parameter',
    transform_function=lambda x: (Transform(Reference(x[1].content), lambda y: ((x[0].content, y[0]),)),))

parameters = AND([
    Discard(open_bracket),
    RepeatingNG(AND([
        OR([
            Transform(name, lambda x: (Discard(match_token('name', x[0].content)),)),
            typed_parameter]),
    ]), 1, transform_function=lambda x: (AND(x),)),
    Discard(close_bracket),
    Discard(arrow),
    Transform(name, lambda x: (x[0].content,)),
], name='parameters')


class Value:
    def get(self):
        raise NotImplementedError


@dataclass
class Constant(Value):
    value: Any

    def get(self):
        return self.value


@dataclass
class Call(Value):
    func: Callable
    args: List[Tuple[str, Value]]

    def get(self):
        return self.func(**{n: v.get() for n, v in self.args})


types: DefaultDict[str, OR] = DefaultDict[str, OR](lambda: OR([]))

types['str'].add_option(match_token('string', tf=lambda x: (Constant(x.content),)))
types['number'].add_option(match_token('number', tf=lambda x: (Constant(x.content),)))
types['name'].add_option(match_token('name', tf=lambda x: (Constant(x.content),)))


def feature(syntax: str) -> Callable[[Callable], Callable]:
    tokens = list(next(lexer.parse(Rewindable(syntax))))
    (parser, return_type), = parameters.parse(Rewindable(tokens))
    parser.resolve_references(types)

    def inner(func: Callable) -> Callable:
        types[return_type].add_option(Transform(parser, lambda x: (Call(func, x),)))
        return func

    return inner


statement = AND([types['void'], Discard(OR([newline, EOI()])), Maybe(Reference('statement'))])
statement.resolve_references({'statement': statement})


def execute(file: str):
    tokens = list(next(lexer.parse(Rewindable(file))))
    tokens = Rewindable(tokens)
    try:
        option, = statement.parse(tokens)
    except ValueError as e:
        print("Parsing failed: ",e)
        return
    for st in option:
        st.get()
