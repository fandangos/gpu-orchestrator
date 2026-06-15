"""Tests for pattern matching."""

from gpu_orchestrator.patterns import matches_gpu, GPU_PATTERNS


def test_comfyui():
    assert matches_gpu("comfyui --port 8188") is not None


def test_stable_diffusion_webui():
    assert matches_gpu("python webui.py") is not None
    assert matches_gpu("stable-diffusion-webui/run.sh") is not None
    assert matches_gpu("automatic1111/stable-diffusion-webui/webui.sh") is not None
    assert matches_gpu("fooocus run") is not None


def test_sd_scripts():
    assert matches_gpu("sd_scripts/train_network.py") is not None


def test_torch_training():
    assert matches_gpu("torchrun --nproc_per_node=2 train.py") is not None
    assert matches_gpu("accelerate launch train.py") is not None
    assert matches_gpu("deepspeed train.py") is not None


def test_inpaint_img2img():
    assert matches_gpu("python inpaint.py image.jpg") is not None
    assert matches_gpu("python img2img.py") is not None
    assert matches_gpu("python txt2img.py") is not None
    assert matches_gpu("python inpaint_net.py") is not None


def test_flux():
    assert matches_gpu("python flux_generate.py") is not None
    assert matches_gpu("flux1 dev") is not None


def test_venv_detection():
    assert matches_gpu("~/.venvs/my-comfyui/run.sh") is not None
    assert matches_gpu("~/.venvs/flux-tool/main.py") is not None


def test_no_match_normal_command():
    assert matches_gpu("echo hello") is None
    assert matches_gpu("ls -la /tmp") is None
    assert matches_gpu("git status") is None
    assert matches_gpu("python normal_script.py") is None
    assert matches_gpu("pip install requests") is None


def test_no_match_with_extra_patterns():
    assert matches_gpu("echo hello", extra_patterns=["custom_pattern"]) is None


def test_case_insensitive():
    assert matches_gpu("COMFYUI run") is not None
    assert matches_gpu("Python Inpaint.py") is not None


def test_custom_extra_patterns():
    result = matches_gpu("my_custom_gpu_tool arg1", extra_patterns=["my_custom_gpu_tool"])
    assert result is not None

    result2 = matches_gpu("my_custom_gpu_tool arg1", extra_patterns=["other_pattern"])
    assert result2 is None


def test_builtin_patterns_list_not_empty():
    assert len(GPU_PATTERNS) > 0


def test_anchored_pattern_safety():
    """Ensure that just mentioning a path in cmdline doesn't trigger a match."""
    # A command that mentions "inpaint.py" as a file path but isn't running it
    assert matches_gpu("cat /tmp/inpaint.py.bak") is None
