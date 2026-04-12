from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


class NeuroScriptError(Exception):
    pass


class ReturnSignal(Exception):
    def __init__(self, value: object) -> None:
        super().__init__()
        self.value = value


@dataclass
class BareValue:
    raw: str


@dataclass
class BinaryExpr:
    left: object
    operator: str
    right: object


@dataclass
class CallExpr:
    name: str
    arguments: list[object]


@dataclass
class Comparison:
    left: object
    operator: str
    right: object


@dataclass
class Assignment:
    name: str
    expression: object


@dataclass
class InputStatement:
    name: str


@dataclass
class OutputStatement:
    expression: object


@dataclass
class InlineIf:
    condition: Comparison
    then_instruction: object
    else_instruction: object | None


@dataclass
class BlockIf:
    condition: Comparison
    then_body: list[object]
    else_body: list[object]


@dataclass
class WhileLoop:
    condition: Comparison
    body: list[object]


@dataclass
class FunctionDefinition:
    name: str
    parameters: list[str]
    body: list[object]


@dataclass
class ReturnStatement:
    expression: object


def normalize_source(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n").lower()


def split_instructions(source: str) -> list[str]:
    instructions: list[str] = []
    buffer: list[str] = []

    for char in source:
        if char == ".":
            instruction = "".join(buffer).strip()
            if instruction:
                instructions.append(" ".join(instruction.split()))
            buffer = []
        else:
            buffer.append(char)

    trailing = "".join(buffer).strip()
    if trailing:
        raise NeuroScriptError("Последняя инструкция должна заканчиваться точкой.")

    return instructions


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []

    for char in text:
        if char in {",", "="}:
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(char)
        elif char.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
        else:
            buffer.append(char)

    if buffer:
        tokens.append("".join(buffer))

    return tokens


class ExpressionParser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0

    def current(self) -> str | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def consume(self, expected: str | None = None) -> str:
        token = self.current()
        if token is None:
            raise NeuroScriptError("Неожиданный конец выражения.")
        if expected is not None and token != expected:
            raise NeuroScriptError(f"Ожидалось {expected}, получено {token}.")
        self.position += 1
        return token

    def parse_expression(self, stop_words: set[str] | None = None, stop_on_comma: bool = False) -> object:
        return self.parse_addition(stop_words or set(), stop_on_comma)

    def parse_addition(self, stop_words: set[str], stop_on_comma: bool) -> object:
        expression = self.parse_multiplication(stop_words, stop_on_comma)

        while True:
            token = self.current()
            if token in {"плюс", "минус"} and token not in stop_words:
                operator = self.consume()
                right = self.parse_multiplication(stop_words, stop_on_comma)
                expression = BinaryExpr(expression, operator, right)
                continue
            return expression

    def parse_multiplication(self, stop_words: set[str], stop_on_comma: bool) -> object:
        expression = self.parse_atom(stop_words, stop_on_comma)

        while True:
            token = self.current()
            if token in {"умножить", "разделить"} and token not in stop_words:
                operator = self.consume()
                right = self.parse_atom(stop_words, stop_on_comma)
                expression = BinaryExpr(expression, operator, right)
                continue
            return expression

    def parse_atom(self, stop_words: set[str], stop_on_comma: bool) -> object:
        token = self.current()
        if token is None:
            raise NeuroScriptError("Ожидалось значение.")

        if token == "вызов" and self.peek("функции"):
            return self.parse_call()

        parts: list[str] = []
        while True:
            token = self.current()
            if token is None:
                break
            if stop_on_comma and token == ",":
                break
            if token in stop_words:
                break
            if token in {"плюс", "минус", "умножить", "разделить"}:
                break
            if token == "=":
                break
            parts.append(self.consume())

        if not parts:
            raise NeuroScriptError("Не удалось разобрать значение.")

        return BareValue(" ".join(parts))

    def parse_call(self) -> CallExpr:
        self.consume("вызов")
        self.consume("функции")
        name = self.consume()
        arguments: list[object] = []

        if self.current() == ",":
            self.consume(",")
            self.consume("аргументы")

            while self.current() is not None:
                arguments.append(self.parse_expression(stop_on_comma=True))
                if self.current() == ",":
                    self.consume(",")
                    continue
                break

        return CallExpr(name, arguments)

    def peek(self, expected: str) -> bool:
        next_index = self.position + 1
        return next_index < len(self.tokens) and self.tokens[next_index] == expected


def parse_expression(text: str) -> object:
    tokens = tokenize(text)
    parser = ExpressionParser(tokens)
    expression = parser.parse_expression()
    if parser.current() is not None:
        raise NeuroScriptError(f"Лишние токены в выражении: {' '.join(tokens[parser.position:])}.")
    return expression


def parse_condition(text: str) -> Comparison:
    tokens = tokenize(text)
    parser = ExpressionParser(tokens)
    comparison_words = {"равно", "больше", "меньше"}
    left = parser.parse_expression(stop_words=comparison_words)
    operator = parser.current()

    if operator not in comparison_words:
        raise NeuroScriptError(f"Не найден оператор сравнения в условии: {text}.")

    parser.consume()
    right = parser.parse_expression()

    if parser.current() is not None:
        raise NeuroScriptError(f"Лишние токены в условии: {' '.join(tokens[parser.position:])}.")

    return Comparison(left, operator, right)


class Parser:
    def __init__(self, instructions: list[str]) -> None:
        self.instructions = instructions

    def parse(self) -> list[object]:
        body, index, stop_word = self.parse_block(0, set())
        if stop_word is not None or index != len(self.instructions):
            raise NeuroScriptError("Не удалось полностью разобрать программу.")
        return body

    def parse_block(self, start_index: int, stop_words: set[str]) -> tuple[list[object], int, str | None]:
        body: list[object] = []
        index = start_index

        while index < len(self.instructions):
            raw = self.instructions[index]
            if raw in stop_words:
                return body, index, raw

            statement, index = self.parse_statement(index)
            body.append(statement)

        return body, index, None

    def parse_statement(self, index: int) -> tuple[object, int]:
        raw = self.instructions[index]

        if raw == "иначе" or raw.startswith("конец "):
            raise NeuroScriptError(f"Неожиданная служебная инструкция: {raw}.")

        if raw.startswith("если "):
            return self.parse_if(index)

        if raw.startswith("пока "):
            condition = parse_condition(raw.removeprefix("пока ").strip())
            body, stop_index, stop_word = self.parse_block(index + 1, {"конец пока"})
            if stop_word != "конец пока":
                raise NeuroScriptError("Для цикла пока не найден конец пока.")
            return WhileLoop(condition, body), stop_index + 1

        if raw.startswith("функция "):
            header_parts = [part.strip() for part in raw.split(",")]
            name = header_parts[0].removeprefix("функция ").strip()
            if not name:
                raise NeuroScriptError("У функции должно быть имя.")

            parameters: list[str] = []
            if len(header_parts) > 1:
                if not header_parts[1].startswith("параметры"):
                    raise NeuroScriptError(f"Неверный заголовок функции: {raw}.")
                first_parameter = header_parts[1].removeprefix("параметры").strip()
                if first_parameter:
                    parameters.append(first_parameter)
                for extra in header_parts[2:]:
                    if extra:
                        parameters.append(extra)

            body, stop_index, stop_word = self.parse_block(index + 1, {"конец функции"})
            if stop_word != "конец функции":
                raise NeuroScriptError("Для функции не найден конец функции.")
            return FunctionDefinition(name, parameters, body), stop_index + 1

        if raw.startswith("вернуть "):
            expression = parse_expression(raw.removeprefix("вернуть ").strip())
            return ReturnStatement(expression), index + 1

        if raw.startswith("ввести "):
            name = raw.removeprefix("ввести ").strip()
            return InputStatement(name), index + 1

        if raw.startswith("вывести "):
            expression = parse_expression(raw.removeprefix("вывести ").strip())
            return OutputStatement(expression), index + 1

        if "=" in raw:
            left, right = raw.split("=", 1)
            name = left.strip()
            expression = parse_expression(right.strip())
            if not name or " " in name:
                raise NeuroScriptError(f"Неверное имя переменной: {left}.")
            return Assignment(name, expression), index + 1

        raise NeuroScriptError(f"Неизвестная инструкция: {raw}.")

    def parse_if(self, index: int) -> tuple[object, int]:
        raw = self.instructions[index]
        condition_text = raw.removeprefix("если ").strip()

        if "," in condition_text:
            first_comma = condition_text.find(",")
            condition = parse_condition(condition_text[:first_comma].strip())
            remainder = condition_text[first_comma + 1 :].strip()

            if ", иначе " in remainder:
                then_raw, else_raw = remainder.split(", иначе ", 1)
                then_instruction = self.parse_inline_instruction(then_raw.strip())
                else_instruction = self.parse_inline_instruction(else_raw.strip())
                return InlineIf(condition, then_instruction, else_instruction), index + 1

            then_instruction = self.parse_inline_instruction(remainder)
            return InlineIf(condition, then_instruction, None), index + 1

        condition = parse_condition(condition_text)
        then_body, stop_index, stop_word = self.parse_block(index + 1, {"иначе", "конец если"})
        if stop_word == "иначе":
            else_body, final_index, final_stop_word = self.parse_block(stop_index + 1, {"конец если"})
            if final_stop_word != "конец если":
                raise NeuroScriptError("Для блока иначе не найден конец если.")
            return BlockIf(condition, then_body, else_body), final_index + 1

        if stop_word != "конец если":
            raise NeuroScriptError("Для условия не найден конец если.")

        return BlockIf(condition, then_body, []), stop_index + 1

    def parse_inline_instruction(self, raw: str) -> object:
        statement, next_index = Parser([raw]).parse_statement(0)
        if next_index != 1:
            raise NeuroScriptError(f"Не удалось разобрать встроенную инструкцию: {raw}.")
        return statement


class Scope:
    def __init__(self, parent: Scope | None = None) -> None:
        self.parent = parent
        self.values: dict[str, object] = {}

    def has(self, name: str) -> bool:
        if name in self.values:
            return True
        return self.parent.has(name) if self.parent is not None else False

    def get(self, name: str) -> object:
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise NeuroScriptError(f"Переменная {name} не определена.")

    def set_local(self, name: str, value: object) -> None:
        self.values[name] = value


class Interpreter:
    def __init__(self) -> None:
        self.functions: dict[str, FunctionDefinition] = {}
        self.global_scope = Scope()

    def run(self, program: list[object]) -> None:
        self.execute_block(program, self.global_scope)

    def execute_block(self, statements: list[object], scope: Scope) -> None:
        for statement in statements:
            self.execute_statement(statement, scope)

    def execute_statement(self, statement: object, scope: Scope) -> None:
        if isinstance(statement, Assignment):
            value = self.evaluate(statement.expression, scope)
            scope.set_local(statement.name, value)
            return

        if isinstance(statement, InputStatement):
            value = input().strip()
            scope.set_local(statement.name, int(value) if value.isdigit() else value)
            return

        if isinstance(statement, OutputStatement):
            value = self.evaluate(statement.expression, scope)
            print(self.stringify(value))
            return

        if isinstance(statement, InlineIf):
            if self.evaluate_condition(statement.condition, scope):
                self.execute_statement(statement.then_instruction, scope)
            elif statement.else_instruction is not None:
                self.execute_statement(statement.else_instruction, scope)
            return

        if isinstance(statement, BlockIf):
            if self.evaluate_condition(statement.condition, scope):
                self.execute_block(statement.then_body, scope)
            else:
                self.execute_block(statement.else_body, scope)
            return

        if isinstance(statement, WhileLoop):
            while self.evaluate_condition(statement.condition, scope):
                self.execute_block(statement.body, scope)
            return

        if isinstance(statement, FunctionDefinition):
            self.functions[statement.name] = statement
            return

        if isinstance(statement, ReturnStatement):
            raise ReturnSignal(self.evaluate(statement.expression, scope))

        raise NeuroScriptError(f"Неизвестный тип инструкции: {type(statement).__name__}.")

    def evaluate_condition(self, condition: Comparison, scope: Scope) -> bool:
        left = self.evaluate(condition.left, scope)
        right = self.evaluate(condition.right, scope)

        if condition.operator == "равно":
            if isinstance(left, str) and isinstance(right, str):
                return left.casefold() == right.casefold()
            return left == right

        self.ensure_number(left, condition.operator)
        self.ensure_number(right, condition.operator)

        if condition.operator == "больше":
            return left > right
        if condition.operator == "меньше":
            return left < right

        raise NeuroScriptError(f"Неизвестный оператор сравнения: {condition.operator}.")

    def evaluate(self, expression: object, scope: Scope) -> object:
        if isinstance(expression, BareValue):
            return self.resolve_bare_value(expression.raw, scope)

        if isinstance(expression, BinaryExpr):
            left = self.evaluate(expression.left, scope)
            right = self.evaluate(expression.right, scope)
            return self.apply_operator(expression.operator, left, right)

        if isinstance(expression, CallExpr):
            if expression.name not in self.functions:
                raise NeuroScriptError(f"Функция {expression.name} не определена.")

            definition = self.functions[expression.name]
            if len(expression.arguments) != len(definition.parameters):
                raise NeuroScriptError(
                    f"Функция {expression.name} ожидает {len(definition.parameters)} аргументов, "
                    f"получено {len(expression.arguments)}."
                )

            function_scope = Scope(self.global_scope)
            for parameter, argument_expression in zip(definition.parameters, expression.arguments):
                function_scope.set_local(parameter, self.evaluate(argument_expression, scope))

            try:
                self.execute_block(definition.body, function_scope)
            except ReturnSignal as signal:
                return signal.value
            return None

        raise NeuroScriptError(f"Неизвестный тип выражения: {type(expression).__name__}.")

    def resolve_bare_value(self, raw: str, scope: Scope) -> object:
        if raw.isdigit():
            return int(raw)

        if " " not in raw and scope.has(raw):
            return scope.get(raw)

        return raw

    def apply_operator(self, operator: str, left: object, right: object) -> object:
        if operator == "плюс":
            if isinstance(left, int) and isinstance(right, int):
                return left + right
            return f"{self.stringify(left)} {self.stringify(right)}".strip()

        self.ensure_number(left, operator)
        self.ensure_number(right, operator)

        if operator == "минус":
            return left - right
        if operator == "умножить":
            return left * right
        if operator == "разделить":
            if right == 0:
                raise NeuroScriptError("Деление на ноль запрещено.")
            return left // right

        raise NeuroScriptError(f"Неизвестный оператор: {operator}.")

    @staticmethod
    def ensure_number(value: object, operator: str) -> None:
        if not isinstance(value, int):
            raise NeuroScriptError(f"Оператор {operator} работает только с числами.")

    @staticmethod
    def stringify(value: object) -> str:
        if value is None:
            return ""
        return str(value)


def run_file(path: Path) -> None:
    if not path.exists():
        raise NeuroScriptError(f"Файл не найден: {path}.")

    source = path.read_text(encoding="utf-8")
    instructions = split_instructions(normalize_source(source))
    program = Parser(instructions).parse()
    Interpreter().run(program)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Использование: python ns.py путь_к_файлу.ns", file=sys.stderr)
        return 1

    target = Path(argv[1]).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target

    try:
        run_file(target.resolve())
    except NeuroScriptError as error:
        print(f"Ошибка NeuroScript: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
