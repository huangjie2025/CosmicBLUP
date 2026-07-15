#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>

namespace cosmic {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERR
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLogFile(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (log_file_.is_open()) {
            log_file_.close();
        }
        log_file_.open(filename, std::ios::out);
        if (!log_file_.is_open()) {
            std::cerr << "Failed to open log file: " << filename << std::endl;
        }
    }

    void setLevel(LogLevel level) {
        level_ = level;
    }

    template<typename T>
    void log(LogLevel level, const T& msg) {
        if (level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        std::stringstream ss;
        if (level == LogLevel::INFO) {
            ss << msg;
        } else {
            ss << getCurrentTime() << " [" << levelToString(level) << "] " << msg;
        }

        std::string log_msg = ss.str();

        // Console output
        if (level >= LogLevel::WARNING) {
            std::cerr << log_msg << std::endl;
        } else {
            std::cout << log_msg << std::endl;
        }

        // File output
        if (log_file_.is_open()) {
            log_file_ << log_msg << std::endl;
        }
    }

    // Helper for formatted logging (printf style)
    template<typename... Args>
    void logFmt(LogLevel level, const char* fmt, Args... args) {
        if (level < level_) return;
        char buffer[1024];
        std::snprintf(buffer, sizeof(buffer), fmt, args...);
        log(level, std::string(buffer));
    }

    // Raw logging (no timestamp/level prefix, useful for headers/tables)
    void logRaw(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << msg; // No newline by default for raw
        if (log_file_.is_open()) {
            log_file_ << msg;
        }
    }

private:
    Logger() : level_(LogLevel::INFO) {}
    ~Logger() {
        if (log_file_.is_open()) log_file_.close();
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::ofstream log_file_;
    LogLevel level_;
    std::mutex mutex_;

    std::string getCurrentTime() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
        #ifdef _WIN32
        localtime_s(&tm_now, &in_time_t);
        #else
        localtime_r(&in_time_t, &tm_now);
        #endif

        std::stringstream ss;
        ss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    const char* levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};

// Global helper macros
#define LOG_INFO(msg) do { std::ostringstream __ss; __ss << msg; ::cosmic::Logger::getInstance().log(::cosmic::LogLevel::INFO, __ss.str()); } while(0)
#define LOG_WARN(msg) do { std::ostringstream __ss; __ss << msg; ::cosmic::Logger::getInstance().log(::cosmic::LogLevel::WARNING, __ss.str()); } while(0)
#define LOG_ERROR(msg) do { std::ostringstream __ss; __ss << msg; ::cosmic::Logger::getInstance().log(::cosmic::LogLevel::ERR, __ss.str()); } while(0)
#define LOG_DEBUG(msg) do { std::ostringstream __ss; __ss << msg; ::cosmic::Logger::getInstance().log(::cosmic::LogLevel::DEBUG, __ss.str()); } while(0)
#define LOG_RAW(msg) ::cosmic::Logger::getInstance().logRaw(msg)

} // namespace cosmic
