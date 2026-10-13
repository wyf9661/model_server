//*****************************************************************************
// Copyright 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ovms {

enum class RequestLifecycleStatus {
    QUEUED,     // accepted by the server, generation has not started yet
    RUNNING,    // first token produced (streaming: exact; unary: inferred, see .cpp)
    COMPLETED,  // generation finished, response delivered to the client
    FAILED,     // terminated with an error before completion
};

struct RequestStatusSnapshot {
    std::string id;
    std::string model;
    uint64_t seqNo = 0;
    uint64_t queuePosition = 0;   // 1-based among in-flight; 0 when not in flight
    int64_t submittedAtUs = 0;  // wall clock, microseconds since epoch
    int64_t scheduledAtUs = 0;  // handed to the generation pipeline; 0 = not yet
    int64_t startedAtUs = 0;    // first token; 0 = unknown yet
    int64_t completedAtUs = 0;  // 0 = still in flight
    RequestLifecycleStatus status = RequestLifecycleStatus::QUEUED;
    int64_t ttftUs = -1;        // exact queue+prefill from genai perf metrics; -1 unknown
    std::string error;
};

// In-process registry of inference requests, backing the per-request status
// query endpoint (GET /v1/requests/{id}). OVMS itself does not expose request
// identifiers or a per-request state machine, so the OpenAI-compatible layer
// records the lifecycle here: every accepted chat/completions or responses
// request is registered when its completion id is generated, and the
// registry is queried by the REST layer.
//
// Status semantics:
//   - QUEUED:    registered, generation not started. queue_position counts
//                in-flight requests registered before this one.
//   - RUNNING:   first token produced. For streaming requests this is exact
//                (first content chunk). For unary requests the CB pipeline
//                gives no mid-flight callback, so the transition is inferred
//                lazily at query time from the pipeline's scheduled-request
//                count; startedAtUs is then the first observation time. The
//                exact start is always recorded at completion from the genai
//                TTFT perf metric (startedAtUs = scheduledAtUs + ttft).
//   - COMPLETED / FAILED: terminal, with exact end timestamp.
//
// Records are kept for a bounded number of most-recent requests
// (terminal records are evicted first; in-flight records are never evicted).
class RequestStatusRegistry {
public:
    static RequestStatusRegistry& instance();

    // Called by the LLM calculator right after request parsing succeeds
    // (the completion id exists from that moment on).
    void registerRequest(const std::string& id, const std::string& model);

    // Called after the request has been handed to the generation pipeline.
    void markScheduled(const std::string& id);

    // Exact start signal (streaming first content chunk).
    void markStarted(const std::string& id);

    // Terminal states. ttftUs: exact queue+prefill from genai perf metrics
    // (>= 0), or -1 when unavailable.
    void markCompleted(const std::string& id, int64_t ttftUs = -1);
    void markFailed(const std::string& id, const std::string& error);

    // Attaches a probe reporting how many requests the request's pipeline
    // currently has scheduled for generation (continuous batching servables).
    // Used to refine QUEUED -> RUNNING lazily at query time for unary
    // requests. Servables without a queue never attach a probe and mark
    // themselves started at schedule time instead.
    void attachScheduledProbe(const std::string& id, std::function<std::optional<size_t>()> probe);

    // Returns nullopt for unknown or evicted ids.
    std::optional<RequestStatusSnapshot> query(const std::string& id);

    // In-flight requests ordered by submission (oldest first).
    std::vector<RequestStatusSnapshot> listInFlight();

    size_t size() const;

private:
    RequestStatusRegistry() = default;
    RequestStatusRegistry(const RequestStatusRegistry&) = delete;
    RequestStatusRegistry& operator=(const RequestStatusRegistry&) = delete;

    struct Record {
        RequestStatusSnapshot snap;
        std::function<std::optional<size_t>()> scheduledProbe;
        // True when startedAtUs was observed exactly (streaming first chunk)
        // rather than inferred (lazy probe refinement at query time). TTFT
        // back-fill at completion only applies to inferred starts.
        bool startedExact = false;
    };

    static int64_t nowUs();
    void refineQueuedLocked();
    void evictOldTerminalLocked();

    mutable std::mutex mtx;
    std::map<uint64_t, Record> records;                 // ordered by seqNo
    std::unordered_map<std::string, uint64_t> idIndex;  // id -> seqNo
    std::atomic<uint64_t> seqCounter{0};

    static constexpr size_t MAX_RECORDS = 4096;
};

}  // namespace ovms
