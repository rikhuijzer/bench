import src.typ
import rasa_nlu.training_data
import typing
import src.training_data


def get_mock_messages() -> typing.Iterable[rasa_nlu.training_data.Message]:
    def create_mock_message(x: int) -> rasa_nlu.training_data.Message:
        return src.training_data.create_message(str(x), 'A' if 0 <= x < 10 else 'B', [], True if x < 15 else False)

    return map(create_mock_message, range(0, 20))


def train(sc: src.typ.SystemCorpus) -> src.typ.System:
    data = list(sc.system.data)
    data[0] += 1
    return src.typ.System(sc.system.name, sc.corpus, tuple(data))


def get_response(query: src.typ.Query) -> src.typ.Response:
    value = int(query.text)
    classification = 'C' if (value % query.system.data[0] == 0) else ('A' if (0 < value <= 10) else 'B')
    return src.typ.Response(classification, 1.0, [])
