#ifndef CHROME_PROFILER_H
#define CHROME_PROFILER_H

#ifdef __cplusplus
#include <chrono>
#include <string>
#include <fstream>
#include <mutex>
#include <vector>
#include <thread>

struct ProfileResult {
    std::string name;
    long long start, end;
    uint32_t threadID;
};

class Profiler {
private:
    std::vector<ProfileResult> m_ProfileHistory;
    std::mutex m_Lock;
    Profiler() { m_ProfileHistory.reserve(100000); }
public:
    static Profiler& Get() {
        static Profiler instance;
        return instance;
    }
    void WriteProfile(const ProfileResult& result) {
        std::lock_guard<std::mutex> lock(m_Lock);
        m_ProfileHistory.push_back(result);
    }
    void Dump(const std::string& filepath) {
        std::ofstream os(filepath);
        os << "[\n";
        for (size_t i = 0; i < m_ProfileHistory.size(); i++) {
            auto& res = m_ProfileHistory[i];
            os << "  {";
            os << "\"cat\":\"function\",";
            os << "\"dur\":" << (res.end - res.start) << ",";
            os << "\"name\":\"" << res.name << "\",";
            os << "\"ph\":\"X\",";
            os << "\"pid\":0,";
            os << "\"tid\":" << res.threadID << ",";
            os << "\"ts\":" << res.start;
            os << "}";
            if (i < m_ProfileHistory.size() - 1) os << ",";
            os << "\n";
        }
        os << "]\n";
        m_ProfileHistory.clear();
    }
};

class ProfileTimer {
private:
    std::string m_Name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
public:
    ProfileTimer(const char* name) : m_Name(name) {
        m_StartTimepoint = std::chrono::high_resolution_clock::now();
    }
    ~ProfileTimer() {
        auto endTimepoint = std::chrono::high_resolution_clock::now();
        long long start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
        long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();
        uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
        Profiler::Get().WriteProfile({m_Name, start, end, threadID});
    }
};

// C++ MACROS
#define PROFILER_DUMP(filepath) Profiler::Get().Dump(filepath)
#define PROFILE_SCOPE(name) ProfileTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

#else
// ==================== C IMPLEMENTATION ====================
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <string.h>

typedef struct {
    char name[64];
    long long start;
    long long end;
    uint32_t threadID;
} CProfileResult;

static CProfileResult c_profile_history[100000];
static int c_profile_count = 0;
static pthread_mutex_t c_profile_lock = PTHREAD_MUTEX_INITIALIZER;

static inline long long get_time_micro() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

static inline void c_profiler_write(const char* name, long long start, long long end) {
    pthread_mutex_lock(&c_profile_lock);
    if (c_profile_count < 100000) {
        strncpy(c_profile_history[c_profile_count].name, name, 63);
        c_profile_history[c_profile_count].name[63] = '\0';
        c_profile_history[c_profile_count].start = start;
        c_profile_history[c_profile_count].end = end;
        c_profile_history[c_profile_count].threadID = (uint32_t)pthread_self();
        c_profile_count++;
    }
    pthread_mutex_unlock(&c_profile_lock);
}

static inline void c_profiler_dump(const char* filepath) {
    FILE* f = fopen(filepath, "w");
    if (!f) return;
    fprintf(f, "[\n");
    for (int i = 0; i < c_profile_count; i++) {
        CProfileResult res = c_profile_history[i];
        fprintf(f, "  {");
        fprintf(f, "\"cat\":\"function\",");
        fprintf(f, "\"dur\":%lld,", res.end - res.start);
        fprintf(f, "\"name\":\"%s\",", res.name);
        fprintf(f, "\"ph\":\"X\",");
        fprintf(f, "\"pid\":0,");
        fprintf(f, "\"tid\":%u,", res.threadID);
        fprintf(f, "\"ts\":%lld", res.start);
        fprintf(f, "}");
        if (i < c_profile_count - 1) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "]\n");
    fclose(f);
    c_profile_count = 0; // reset for next run
}

// C MACROS
#define PROFILER_DUMP(filepath) c_profiler_dump(filepath)
#define PROFILE_START(name) long long _prof_start_##name = get_time_micro()
#define PROFILE_END(name) c_profiler_write(#name, _prof_start_##name, get_time_micro())

#endif

#endif
