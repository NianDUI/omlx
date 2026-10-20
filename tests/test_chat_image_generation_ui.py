"""Regression guards for local image generation in the standalone web app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT_TEMPLATE = ROOT / "apps/omlx-web/omlx_web/templates/chat.html"

def _template() -> str:
    return CHAT_TEMPLATE.read_text()


def _section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_image_generation_models_use_image_endpoint_and_render_results():
    html = _template()
    send = _section(html, "    async sendMessage()", "    async generateImageResponse(imageContext)")
    generate = _section(
        html,
        "    async generateImageResponse(imageContext)",
        "async streamResponse(streamContext = null, depth = 0)",
    )
    assistant_renderer = _section(
        html,
        "<!-- Assistant Message -->",
        "<!-- Streaming Message -->",
    )

    assert "this.isImageGenerationModel(this.currentModel)" in send
    assert "await this.generateImageResponse" in send
    assert "!this.isImageGenerationModel(this.currentModel)" in html
    assert "fetch('/v1/images/generations'" in generate
    assert "response_format: 'b64_json'" in generate
    assert "...imageSettings" in generate
    assert "data:image/png;base64,${item.b64_json}" in generate
    assert 'getImageUrls(msg.content).length > 0' in assistant_renderer
    assert 'renderMarkdown(getTextContent(msg.content))' in assistant_renderer
    assert 'renderMarkdown(msg.content)' not in assistant_renderer


def test_image_generation_regeneration_stays_on_image_endpoint():
    regenerate = _section(
        _template(),
        "    async regenerateMessage(index, opts = {})",
        "_copyFallback(text)",
    )

    assert "this.isImageGenerationModel(targetModel)" in regenerate
    assert "await this.generateImageResponse" in regenerate
    assert "this.getTextContent(session.messages[userIdx]?.content).trim()" in regenerate


def test_image_generation_settings_are_editable_and_persisted_per_model():
    html = _template()
    generate = _section(
        html,
        "    async generateImageResponse(imageContext)",
        "async streamResponse(streamContext = null, depth = 0)",
    )

    assert 'x-show="isImageGenerationModel(currentModel)"' in html
    assert 'x-model="modelSettings.image_size"' in html
    assert 'x-model.number="modelSettings.image_steps"' in html
    assert 'x-model.number="modelSettings.image_seed"' in html
    assert 'x-model.number="modelSettings.image_guidance"' in html
    assert "'is-on': modelSettings.image_use_context" in html
    assert "modelSettings.image_use_context = !modelSettings.image_use_context" in html
    assert 'x-model.number="modelSettings.image_context_turns"' in html
    assert "resolveImageGenerationSettings()" in html
    assert "const imageSettings = this.resolveImageGenerationSettings()" in generate
    assert "captureGenerationContext(" in generate


def test_image_generation_text_context_is_opt_in_and_user_only():
    html = _template()
    helper = _section(
        html,
        "            buildImageGenerationPrompt(messages, userIndex, fallbackPrompt)",
        "            thinkingModeValue()",
    )
    generate = _section(
        html,
        "    async generateImageResponse(imageContext)",
        "async streamResponse(streamContext = null, depth = 0)",
    )

    assert "image_use_context: false" in html
    assert "image_context_turns: 3" in html
    assert "message?.role !== 'user'" in helper
    assert "prompts.slice(-contextSettings.turns)" in helper
    assert "if (!contextSettings.enabled) return fallback" in helper
    assert "const requestPrompt = this.buildImageGenerationPrompt(" in generate
    assert "prompt: requestPrompt" in generate
    assert "image_use_context: imageContextSettings.enabled" in generate


def test_chat_images_are_persisted_in_indexeddb_and_hydrated_after_reload():
    html = _template()
    save = _section(html, "    saveCurrentChat(", "    startRenamingChat(chat)")
    load = _section(html, "    async loadChat(chatId)", "    saveCurrentChat(")

    assert "CHAT_MEDIA_DB_NAME = 'omlx_chat_media'" in html
    assert "openChatMediaDb()" in html
    assert "prepareMessagesForStorage(chatId, messages)" in html
    assert "persistChatMediaForMessages(chatId, messages" in html
    assert "media_key: mediaKey" in html
    assert "this.prepareMessagesForStorage(chatId, messagesToSave)" in save
    assert "await this.hydrateChatMedia(chatId, session.messages)" in load
    assert "await this.deleteChatMedia(chatId)" in html
    assert "await this.clearChatMedia()" in html


def test_chat_media_lifecycle_waits_for_writes_and_cleans_orphans():
    html = _template()
    send = _section(html, "    async sendMessage()", "    async generateImageResponse(imageContext)")
    generate = _section(
        html,
        "    async generateImageResponse(imageContext)",
        "async streamResponse(streamContext = null, depth = 0)",
    )
    branch = _section(html, "    async branchChat(index)", "    computeTimelineDots()")
    delete_chat = _section(html, "    async deleteChat(chatId", "    // Clear all history")

    assert "await this.persistChatMediaForMessages(chatId, [userMsg])" in send
    assert "await this.persistChatMediaForMessages(context.chatId, [assistantMsg])" in generate
    assert "{ forceNewKeys: true }" in branch
    assert "await this.deleteChatMedia(chatId)" in delete_chat
    assert "referencedChatMediaKeys()" in html
    assert "async garbageCollectChatMedia()" in html
    assert "await this.garbageCollectChatMedia()" in html
    assert "validChatIds.has(chatId)" in html
    assert "|| this.isChatStreaming(chatId)" in html

    save_edit = _section(html, "async saveEdit(index)", "// Delete chat")
    assert "await this.garbageCollectChatMedia()" in save_edit
