import json
from unittest.mock import Mock

import pytest
import xgrammar as xgr
from pydantic import BaseModel

from format_out.impl import ChatSession, SamplingParams
from lightllm.server.core.objs.sampling_params import SamplingParams as ServerSamplingParams


class HelperResult(BaseModel):
    answer: str


@pytest.mark.parametrize("mode", ["plain", "regex", "grammar", "json", "inherited_json"])
def test_helper_payload_initializes_server_constraints(monkeypatch, mode):
    payloads = []

    def post(url, headers, data):
        payload = json.loads(data)
        server_params = ServerSamplingParams()
        server_params.init(None, **payload["parameters"])
        assert server_params.constraint_out_params.has_constraint() == (mode != "plain")
        assert all(value is not None for value in payload["parameters"].values())
        payloads.append(payload)
        return Mock(status_code=200, json=lambda: {"generated_text": ['{"answer":"OK"}']})

    monkeypatch.setattr("format_out.impl.requests.post", post)
    params = SamplingParams(do_sample=False, guided_json={} if mode == "inherited_json" else None)
    session = ChatSession("Answer briefly.", params, disable_log=True)
    if mode == "regex":
        session.generate(regex="OK")
        assert payloads[0]["parameters"]["regular_constraint"] == "OK"
    elif mode == "grammar":
        session.generate(grammar=xgr.Grammar.from_ebnf('root ::= "OK"'))
        assert payloads[0]["parameters"]["guided_grammar"]
    elif mode == "json":
        output = session.gen_json_object(HelperResult, prefix_regex=r"\s{0,20}")
        assert HelperResult.model_validate_json(output).answer == "OK"
    else:
        session.generate()
        if mode == "inherited_json":
            assert payloads[0]["parameters"]["guided_json"] == {}
    assert params.regular_constraint is None
    assert params.guided_grammar is None
    assert params.guided_json == ({} if mode == "inherited_json" else None)
