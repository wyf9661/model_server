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
#include "request_status_registry.hpp"

#include <chrono>

namespace ovms {

RequestStatusRegistry& RequestStatusRegistry::instance() {
    static RequestStatusRegistry registry;
    return registry;
}

int64_t RequestStatusRegistry::nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void RequestStatusRegistry::registerRequest(const std::string& id, const std::string& model) {
    std::lock_guard<std::mutex> lock(mtx);
    if (idIndex.count(id))
        return;  // defensive: ids are unique per request by construction
    evictOldTerminalLocked();
    uint64_t seq = ++seqCounter;
    Record record;
    record.snap.id = id;
    record.snap.model = model;
    record.snap.seqNo = seq;
    record.snap.submittedAtUs = nowUs();
    record.snap.status = RequestLifecycleStatus::QUEUED;
    records.emplace(seq, std::move(record));
    idIndex.emplace(id, seq);
}

void RequestStatusRegistry::markScheduled(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = idIndex.find(id);
    if (it == idIndex.end())
        return;
    auto& rec = records.at(it->second);
    if (rec.snap.scheduledAtUs == 0)
        rec.snap.scheduledAtUs = nowUs();
    // No pipeline probe attached: queue-less servable (legacy pipelines).
    // Generation starts immediately once scheduled, so promote to RUNNING.
    if (!rec.scheduledProbe && rec.snap.startedAtUs == 0 && rec.snap.completedAtUs == 0) {
        rec.snap.startedAtUs = nowUs();
        rec.snap.status = RequestLifecycleStatus::RUNNING;
    }
}

void RequestStatusRegistry::markStarted(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = idIndex.find(id);
    if (it == idIndex.end())
        return;
    auto& rec = records.at(it->second);
    if (rec.snap.startedAtUs == 0 && rec.snap.completedAtUs == 0) {
        rec.snap.startedAtUs = nowUs();
        rec.snap.status = RequestLifecycleStatus::RUNNING;
        rec.startedExact = true;
    }
}

void RequestStatusRegistry::markCompleted(const std::string& id, int64_t ttftUs) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = idIndex.find(id);
    if (it == idIndex.end())
        return;
    auto& rec = records.at(it->second);
    if (rec.snap.completedAtUs != 0)
        return;  // already terminal
    // Exact start when genai reports TTFT: the metric measures from pipeline
    // submission to first token, so start = scheduled + ttft. Only applied to
    // inferred starts — an exactly observed start (streaming first chunk) wins.
    if (!rec.startedExact && ttftUs >= 0 && rec.snap.scheduledAtUs > 0) {
        rec.snap.startedAtUs = rec.snap.scheduledAtUs + ttftUs;
        rec.snap.ttftUs = ttftUs;
    } else if (ttftUs >= 0) {
        rec.snap.ttftUs = ttftUs;
    }
    if (rec.snap.startedAtUs == 0)
        rec.snap.startedAtUs = nowUs();
    rec.snap.completedAtUs = nowUs();
    rec.snap.status = RequestLifecycleStatus::COMPLETED;
}

void RequestStatusRegistry::markFailed(const std::string& id, const std::string& error) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = idIndex.find(id);
    if (it == idIndex.end())
        return;
    auto& rec = records.at(it->second);
    if (rec.snap.completedAtUs != 0)
        return;  // already terminal
    rec.snap.completedAtUs = nowUs();
    rec.snap.status = RequestLifecycleStatus::FAILED;
    rec.snap.error = error;
}

void RequestStatusRegistry::attachScheduledProbe(const std::string& id, std::function<std::optional<size_t>()> probe) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = idIndex.find(id);
    if (it == idIndex.end())
        return;
    auto& rec = records.at(it->second);
    rec.scheduledProbe = std::move(probe);
}

// Refines QUEUED -> RUNNING for unary in-flight requests using pipeline
// probes. Called with the registry lock held (query/list paths); genai's
// get_metrics() is a lightweight scheduler snapshot with no callback into
// this registry, so holding the lock across the probe is safe.
void RequestStatusRegistry::refineQueuedLocked() {
    for (auto& [seq, rec] : records) {
        if (rec.snap.status != RequestLifecycleStatus::QUEUED || rec.snap.completedAtUs != 0)
            continue;
        if (!rec.scheduledProbe)
            continue;
        std::optional<size_t> scheduled;
        try {
            scheduled = rec.scheduledProbe();
        } catch (...) {
            continue;  // probe failure: leave the request as QUEUED
        }
        if (scheduled && *scheduled > 0) {
            rec.snap.status = RequestLifecycleStatus::RUNNING;
            if (rec.snap.startedAtUs == 0)
                rec.snap.startedAtUs = nowUs();  // first observation as running
        }
    }
}

std::optional<RequestStatusSnapshot> RequestStatusRegistry::query(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = idIndex.find(id);
    if (it == idIndex.end())
        return std::nullopt;
    refineQueuedLocked();
    auto& rec = records.at(it->second);
    RequestStatusSnapshot snap = rec.snap;

    // Queue position: 1-based index among in-flight requests ordered by
    // submission. Only meaningful while the request is still in flight.
    if (snap.completedAtUs == 0) {
        uint64_t position = 0;
        for (const auto& [seq, other] : records) {
            if (other.snap.completedAtUs == 0)
                position++;
            if (seq == it->second)
                break;
        }
        snap.queuePosition = position;
    }
    return snap;
}

std::vector<RequestStatusSnapshot> RequestStatusRegistry::listInFlight() {
    std::lock_guard<std::mutex> lock(mtx);
    refineQueuedLocked();
    std::vector<RequestStatusSnapshot> out;
    for (const auto& [seq, rec] : records) {
        if (rec.snap.completedAtUs == 0)
            out.push_back(rec.snap);
    }
    return out;
}

size_t RequestStatusRegistry::size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return records.size();
}

void RequestStatusRegistry::evictOldTerminalLocked() {
    if (records.size() < MAX_RECORDS)
        return;
    // Evict oldest terminal records (in-flight records are never evicted).
    for (auto it = records.begin(); it != records.end() && records.size() >= MAX_RECORDS;) {
        if (it->second.snap.completedAtUs != 0) {
            idIndex.erase(it->second.snap.id);
            it = records.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace ovms
