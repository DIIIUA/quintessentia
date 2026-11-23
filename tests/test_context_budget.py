from types import SimpleNamespace
import pytest
from aircolibri_cli.unified_runtime import AirColibriUnifiedEngine

@pytest.mark.parametrize("reservation,native,prompt,output",[(8192,32768,7287,32768),(4096,40960,7287,40000),(16,12,1_000_000,1_000_000)])
def test_no_server_context_ceiling(reservation,native,prompt,output):
    engine=AirColibriUnifiedEngine.__new__(AirColibriUnifiedEngine)
    engine.engine=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=native))
    engine.plan=SimpleNamespace(max_context_tokens=reservation)
    engine._check_context_budget(prompt,output)

def test_invalid_negative_counts_are_still_rejected():
    engine=AirColibriUnifiedEngine.__new__(AirColibriUnifiedEngine)
    with pytest.raises(ValueError):engine._check_context_budget(-1,100)
