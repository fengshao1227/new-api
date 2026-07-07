package relay

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/QuantumNous/new-api/common"
	appconstant "github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/dto"
	relaycommon "github.com/QuantumNous/new-api/relay/common"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newResponsesPromptTestContext(t *testing.T) *gin.Context {
	t.Helper()
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	return c
}

func newResponsesPromptRelayInfo(prompt string, override bool) *relaycommon.RelayInfo {
	return &relaycommon.RelayInfo{
		ChannelMeta: &relaycommon.ChannelMeta{
			ChannelSetting: dto.ChannelSettings{
				SystemPrompt:         prompt,
				SystemPromptOverride: override,
			},
		},
	}
}

// Regression tests for #5961: the channel-level system prompt was applied on
// /v1/chat/completions but silently skipped on /v1/responses.
func TestResponsesSystemPromptSetWhenInstructionsAbsent(t *testing.T) {
	c := newResponsesPromptTestContext(t)
	request := &dto.OpenAIResponsesRequest{Model: "gpt-5.2"}

	applyResponsesSystemPromptIfNeeded(c, newResponsesPromptRelayInfo("channel prompt", false), request)

	assert.JSONEq(t, `"channel prompt"`, string(request.Instructions))
	assert.False(t, common.GetContextKeyBool(c, appconstant.ContextKeySystemPromptOverride))
}

func TestResponsesSystemPromptKeepsClientInstructionsWithoutOverride(t *testing.T) {
	c := newResponsesPromptTestContext(t)
	request := &dto.OpenAIResponsesRequest{
		Model:        "gpt-5.2",
		Instructions: json.RawMessage(`"client instructions"`),
	}

	applyResponsesSystemPromptIfNeeded(c, newResponsesPromptRelayInfo("channel prompt", false), request)

	assert.JSONEq(t, `"client instructions"`, string(request.Instructions))
	assert.False(t, common.GetContextKeyBool(c, appconstant.ContextKeySystemPromptOverride))
}

func TestResponsesSystemPromptPrependsWhenOverrideEnabled(t *testing.T) {
	c := newResponsesPromptTestContext(t)
	request := &dto.OpenAIResponsesRequest{
		Model:        "gpt-5.2",
		Instructions: json.RawMessage(`"client instructions"`),
	}

	applyResponsesSystemPromptIfNeeded(c, newResponsesPromptRelayInfo("channel prompt", true), request)

	var merged string
	require.NoError(t, common.Unmarshal(request.Instructions, &merged))
	assert.Equal(t, "channel prompt\nclient instructions", merged)
	assert.True(t, common.GetContextKeyBool(c, appconstant.ContextKeySystemPromptOverride))
}

func TestResponsesSystemPromptTreatsNullInstructionsAsAbsent(t *testing.T) {
	c := newResponsesPromptTestContext(t)
	request := &dto.OpenAIResponsesRequest{
		Model:        "gpt-5.2",
		Instructions: json.RawMessage(`null`),
	}

	applyResponsesSystemPromptIfNeeded(c, newResponsesPromptRelayInfo("channel prompt", false), request)

	assert.JSONEq(t, `"channel prompt"`, string(request.Instructions))
}

func TestResponsesSystemPromptLeavesNonStringInstructionsUntouched(t *testing.T) {
	c := newResponsesPromptTestContext(t)
	original := json.RawMessage(`[{"type":"text","text":"structured"}]`)
	request := &dto.OpenAIResponsesRequest{
		Model:        "gpt-5.2",
		Instructions: original,
	}

	applyResponsesSystemPromptIfNeeded(c, newResponsesPromptRelayInfo("channel prompt", true), request)

	assert.Equal(t, string(original), string(request.Instructions))
	assert.False(t, common.GetContextKeyBool(c, appconstant.ContextKeySystemPromptOverride))
}

func TestResponsesSystemPromptNoopWithoutChannelPrompt(t *testing.T) {
	c := newResponsesPromptTestContext(t)
	request := &dto.OpenAIResponsesRequest{
		Model:        "gpt-5.2",
		Instructions: json.RawMessage(`"client instructions"`),
	}

	applyResponsesSystemPromptIfNeeded(c, newResponsesPromptRelayInfo("", true), request)

	assert.JSONEq(t, `"client instructions"`, string(request.Instructions))
}
