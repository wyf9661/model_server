//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#pragma once

#include <string>

#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"
#include "status.hpp"

namespace ovms {

// Maps an OVMS status code onto the (type, code) pair used by the OpenAI error object.
// OpenAI clients and gateways key their retry/backoff decisions off these fields, so a
// bare error string is not a drop-in substitute.
struct OpenAIErrorMapping {
    const char* type;
    const char* code;
};

inline const OpenAIErrorMapping openAIErrorMapping(StatusCode statusCode) {
    switch (statusCode) {
        case StatusCode::UNAUTHORIZED:
            return {"invalid_request_error", "invalid_api_key"};
        case StatusCode::MODEL_MISSING:
        case StatusCode::MODEL_NAME_MISSING:
        case StatusCode::MODEL_VERSION_MISSING:
        case StatusCode::MODEL_VERSION_NOT_LOADED_YET:
        case StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE:
        case StatusCode::MODEL_NOT_LOADED:
        case StatusCode::MODELINSTANCE_NOT_FOUND:
        case StatusCode::PIPELINE_DEFINITION_NAME_MISSING:
        case StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET:
        case StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE:
        case StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING:
        case StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE:
            return {"invalid_request_error", "model_not_found"};
        case StatusCode::JSON_INVALID:
        case StatusCode::REST_BODY_IS_NOT_AN_OBJECT:
        case StatusCode::REST_INVALID_URL:
        case StatusCode::REST_NOT_FOUND:
        case StatusCode::REST_COULD_NOT_PARSE_PARAMETERS:
        case StatusCode::INVALID_CONTENT_SIZE:
            return {"invalid_request_error", "invalid_request_error"};
        case StatusCode::MEDIAPIPE_EXECUTION_ERROR:
        case StatusCode::MEDIAPIPE_PRECONDITION_FAILED:
        case StatusCode::REST_COULD_NOT_PARSE_INPUT:
        case StatusCode::REST_COULD_NOT_PARSE_OUTPUT:
            return {"invalid_request_error", "invalid_parameter"};
        case StatusCode::SERVER_NOT_READY:
        case StatusCode::SERVER_SHUTTING_DOWN:
            return {"server_error", "server_unavailable"};
        default:
            return {"server_error", "internal_error"};
    }
}

// Serializes a message + status code as an OpenAI-compatible error object:
//   {"error": {"message": ..., "type": ..., "param": null, "code": ...}}
// OpenAI SDKs read error.message/type/code; the previous bare-string form
// ({"error": "<string>"}) breaks their error handling and retry classification.
inline std::string makeOpenAIErrorBody(const std::string& message, StatusCode statusCode) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    const OpenAIErrorMapping mapping = openAIErrorMapping(statusCode);
    writer.StartObject();
    writer.String("error");
    writer.StartObject();
    writer.String("message");
    writer.String(message.c_str());
    writer.String("type");
    writer.String(mapping.type);
    writer.String("param");
    writer.Null();
    writer.String("code");
    writer.String(mapping.code);
    writer.EndObject();
    writer.EndObject();
    return buffer.GetString();
}

inline std::string makeOpenAIErrorBody(const ovms::Status& status) {
    return makeOpenAIErrorBody(status.string(), status.getCode());
}

}  // namespace ovms