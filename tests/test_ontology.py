from fiesl.ontology import FROZEN_LLM_OUTPUT, audit_agent_output, build_agent_prompt, candidate_manifest, compile_ontology, unit_membership

def test_frozen_ontology_contract() -> None:
    manifest = compile_ontology()
    membership = unit_membership(manifest)
    assert manifest["audit"]["status"] == "PASS"
    assert manifest["unit_count"] == 9
    assert membership.shape == (9, 10)
    assert membership.sum(dim=0).tolist() == [1.0] * 10
    assert membership[2].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_agent_prompt_and_frozen_replay_contract() -> None:
    prompt = build_agent_prompt()
    candidate = candidate_manifest(FROZEN_LLM_OUTPUT)
    assert "TwiBot-20" in prompt and "TwiBot-22" in prompt
    assert "profile.screen_name" in prompt and "user.username" in prompt
    assert audit_agent_output(FROZEN_LLM_OUTPUT)["status"] == "PASS"
    assert candidate["audit"]["status"] == "PASS"
    assert candidate["candidate_only"] is True
