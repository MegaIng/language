from decimal import Decimal

from language import feature


@feature("[replace {old: str} with {new: str} in {target: str}] -> str")
@feature("[in {target: str} replace {old: str} with {new: str}] -> str")
@feature("[replace {old: str} with {new: str} in {target: str} {count: number} times] -> str")
@feature("[in {target: str} replace {old: str} with {new: str} {count: number} times] -> str")
def replace(*, target: str, old: str, new: str, count: int = None) -> str:
    if count is None:
        return target.replace(old, new)
    else:
        return target.replace(old, new, int(count))


@feature("[print {data: str}] -> void")
def print_(*, data: str):
    print(data)


variables = {}


@feature("[store string {data: str} as {variable: name}] -> void")
@feature("[store number {data: number} as {variable: name}] -> void")
def store(*, data, variable: str):
    variables[variable] = data


@feature("[{variable: name}] -> str")
@feature("[{variable: name}] -> number")
def load(*, variable):
    return variables[variable]


@feature("[{data: number}] -> str")
def convert_to_string(*, data) -> str:
    return str(data)


@feature("[add {a: number} and {b: number}] -> number")
def add(*, a: Decimal, b: Decimal) -> Decimal:
    return a + b
