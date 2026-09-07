from smolagents import (
        InferenceClientModel,
        CodeAgent,
        WebSearchTool,         
    )
from weavemuse.tools.stable_audio_tool import StableAudioTool, RemoteStableAudioTool
from weavemuse.tools.audio_analysis_tool import AudioAnalysisTool
from weavemuse.tools.notagen_tool import NotaGenTool, RemoteNotaGenTool
from weavemuse.tools.chat_musician_tool import ChatMusicianTool
from weavemuse.tools.audio_flamingo_tool import AudioFlamingoTool
import os
import warnings


def create_web_agent(model):
    web_agent = CodeAgent(
        tools=[WebSearchTool()],
        model=model,
        name="web_search_agent",
        description="Runs web searches for you. Give it your query as an argument.",
        additional_authorized_imports=["os", "pathlib", "tempfile"]
    )    
    return web_agent


def create_symbolic_music_agent(model, device_map="auto", output_dir="/tmp/notagen_output", remote_only=False):
    # Create NotaGen tool for symbolic music generation
    if remote_only:
        notagen_tool = RemoteNotaGenTool()
    else:
        notagen_tool = NotaGenTool(device=device_map, output_dir=output_dir)

    symbolic_music_agent = CodeAgent(
        tools=[notagen_tool],
        model=model,
        name="symbolic_music_agent",
        description="Generates/Composes symbolic music in ABC notation format with full conversion capabilities. Can create compositions based on musical periods, composers, and instrumentation. Returns a PDF, XML, MIDI, and MP3 of the score. Example usage of this tool: Compose a piano piece in the style of Chopin.",
        max_steps=1
    )  
    return symbolic_music_agent


def create_audio_analysis_agent(model, device_map="auto", remote_only=False):
    # Create Audio Flamingo tool for advanced audio analysis via Gradio Client
    audio_flamingo_tool = AudioFlamingoTool()
    if not remote_only:
        # Create Audio Analysis tool (commented out to save VRAM)      
        audio_analysis_tool = AudioAnalysisTool(device=device_map)
        tl = [audio_flamingo_tool, audio_analysis_tool]
    else:
        tl = [audio_flamingo_tool]
    # Create Audio Flamingo agent for advanced audio analysis
    audio_analysis_agent = CodeAgent(
        tools=tl,
        model=model,
        name="audio_analysis_agent",
        description=(
            "Analyzes audio files using NVIDIA's Audio Flamingo model or other audio analysis tools. "
            "Can answer questions about audio content, describe musical elements, "
            "identify sounds, and provide detailed acoustic analysis. "
            "IMPORTANT: When users upload audio files, use the exact file path provided"
            "without checking if the file exists first. The tool will handle file validation internally."
            "Use first the Audio Flamingo tool, and if it fails, fall back to the Audio Analysis tool."
        ),
        additional_authorized_imports=["gradio_client", "os", "pathlib", "tempfile", "shutil", "posixpath"],
        max_steps=2
    )
    return audio_analysis_agent


def create_audio_generation_agent(model, device_map="auto", output_dir="/tmp/stable_audio", remote_only=False):
    # Create Audio Generation and Analysis agent (temporarily disabled to focus on NotaGen and ChatMusician)
    if remote_only:
        stable_audio_tool = RemoteStableAudioTool()
    else:
        stable_audio_tool = StableAudioTool(device=device_map, output_dir=output_dir)
    audio_generation_agent = CodeAgent(
        tools=[stable_audio_tool],
        model=model,
        name="audio_generation_agent",
        description="Generates audio from text descriptions. Use the 'prompt' argument to specify what you want to hear. Generates up to 47 seconds of high-quality audio.",
    )   
    return audio_generation_agent


def create_musicology_agent(model, data_dir=None):
    """Build the Didone musicology-analysis agent, or return None if the corpus
    data isn't available.

    This agent answers analytical questions about the Didone corpus (18th-c.
    Italian opera arias) from pre-computed structured data via read-only CSV
    lookup tools -- no GPU, no audio, no model weights. Its tools return
    evidence (chord progressions, tonal plans, section structure), not
    conclusions: cadence identification, style judgements and cross-corpus
    norms are left to the agent's own reasoning.
    """
    from weavemuse.tools.didone_tools import didone_tools

    resolved = data_dir or os.getenv("DIDONE_DATA_DIR")
    # cheap availability check so a missing corpus degrades to "agent absent"
    # rather than a hard failure when the manager routes to it
    probe_root = resolved or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if not os.path.exists(os.path.join(probe_root, "metadata.csv")):
        warnings.warn(
            f"musicology_analysis_agent not created: metadata.csv not found under "
            f"{probe_root!r}. Set DIDONE_DATA_DIR to enable it."
        )
        return None
    if resolved:
        os.environ["DIDONE_DATA_DIR"] = resolved

    return CodeAgent(
        tools=didone_tools(),
        model=model,
        name="musicology_analysis_agent",
        description=(
            "Answers analytical questions about the Didone corpus of 18th-century "
            "Italian opera arias (multiple composers setting the same libretto texts, "
            "c. 1720-1800) using pre-computed structured data: score metadata, tonal "
            "plans, chord-by-chord Roman-numeral harmony, and formal/text section "
            "structure. Use it for questions about modulation, cadences, harmonic "
            "rhythm, tonal design, strophic form, style (galant vs Baroque, "
            "stormy vs pastoral), and how one aria compares to others of its period "
            "or composer. Pass the full question plus any known record_id. The agent "
            "looks up evidence and reasons about it -- it does not play or synthesize "
            "audio."
        ),
        additional_authorized_imports=["statistics", "collections", "json", "re", "math"],
        max_steps=12,
    )


def get_weavemuse_agents_and_tools(model=None, device_map="auto", notagen_output_dir="/tmp/notagen_output", stable_audio_output_dir="/tmp/stable_audio", tool_mode="hybrid", include_musicology_agent=True, exclude_agents=None):
    """
    Returns all WeaveMuse agents and tools as a list for easy access and management.

    Args:
        model: The language model to be used by the agents.
        device_map (str): Device mapping for model deployment (default is "auto").
        notagen_output_dir (str): Output directory for NotaGen tool (default is "/tmp/notagen_output").
        stable_audio_output_dir (str): Output directory for Stable Audio tool (default
        include_musicology_agent (bool): build the Didone musicology-analysis agent
            (skipped anyway if its corpus data can't be found).
        exclude_agents (Iterable[str] | None): names to leave out entirely -- neither
            constructed nor returned. Recognised: "web_search_agent",
            "symbolic_music_agent", "audio_analysis_agent", "audio_generation_agent",
            "musicology_analysis_agent", and "chat_musician" (the top-level tool).
            Use this to run the manager with only a subset of capabilities, e.g.
            exclude the two generative agents for an analysis-only study.
    """
    # If model is not provided, load a default InferenceClient model
    if model is None:
        # Load the default InferenceClient model, but produce warning if not specified
        warnings.warn("No model specified, using default InferenceClientModel.")
        model = InferenceClientModel(
            model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            # nebius no longer serves this model (see gui.py's set_up_agents for details)
            provider="featherless-ai"
        )
    excluded = set(exclude_agents or ())
    remote_only = (tool_mode == "remote")

    tools = []
    if "chat_musician" not in excluded and not remote_only:
        tools.append(ChatMusicianTool(device=device_map))

    agents = []
    if "symbolic_music_agent" not in excluded:
        agents.append(create_symbolic_music_agent(model, device_map=device_map, output_dir=notagen_output_dir, remote_only=remote_only))
    if "audio_analysis_agent" not in excluded:
        agents.append(create_audio_analysis_agent(model, device_map=device_map, remote_only=remote_only))
    if "audio_generation_agent" not in excluded:
        agents.append(create_audio_generation_agent(model, device_map=device_map, output_dir=stable_audio_output_dir, remote_only=remote_only))
    if "web_search_agent" not in excluded:
        agents.append(create_web_agent(model))
    if include_musicology_agent and "musicology_analysis_agent" not in excluded:
        musicology_agent = create_musicology_agent(model)
        if musicology_agent is not None:
            agents.append(musicology_agent)
    return agents, tools