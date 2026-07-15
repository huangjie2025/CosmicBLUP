#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cosmic {

// A simple read-only memory mapped file wrapper
class MMapFile {
private:
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = NULL;
#else
    int fd = -1;
#endif
    void* data = nullptr;
    size_t size = 0;

public:
    MMapFile() {}
    ~MMapFile() { close(); }

    void open(const std::string& path) {
        close(); // Close existing if any

#ifdef _WIN32
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Could not open file for mmap: " + path);
        }

        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile, &fileSize)) {
            CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Could not get file size: " + path);
        }
        size = (size_t)fileSize.QuadPart;

        if (size == 0) {
            // Empty file, nothing to map
            return;
        }

        hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMap == NULL) {
            CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Could not create file mapping: " + path);
        }

        data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (data == NULL) {
            CloseHandle(hMap); hMap = NULL;
            CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Could not map view of file: " + path);
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Could not open file for mmap: " + path);
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            ::close(fd); fd = -1;
            throw std::runtime_error("Could not get file size: " + path);
        }
        size = (size_t)sb.st_size;

        if (size == 0) return;

        data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            ::close(fd); fd = -1;
            data = nullptr;
            throw std::runtime_error("Could not mmap file: " + path);
        }
#endif
    }

    void close() {
#ifdef _WIN32
        if (data) { UnmapViewOfFile(data); data = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = NULL; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
#else
        if (data) { munmap(data, size); data = nullptr; }
        if (fd != -1) { ::close(fd); fd = -1; }
#endif
        size = 0;
    }

    const void* get_data() const { return data; }
    size_t get_size() const { return size; }
    bool is_open() const { return data != nullptr; }
};

// Wrapper for Triplet access from MMap
// Binary format assumed: int32(row), int32(col), float(val) - 12 bytes per record
struct MMapTriplet {
    int row;
    int col;
    double val;
};

class MMapMatrix {
private:
    MMapFile mmap;
    size_t n_entries = 0;
    int _rows = 0;
    int _cols = 0;

public:
    MMapMatrix() {}

    void open(const std::string& path) {
        mmap.open(path);
        size_t record_size = 12; // 4+4+4
        if (mmap.get_size() % record_size != 0) {
            std::cerr << "Warning: MMap file size not multiple of 12 bytes. " << path << "\n";
        }
        n_entries = mmap.get_size() / record_size;

        // Scan for max dim?
        // This defeats the purpose of fast load if we scan entire file.
        // But we need dimensions.
        // Option 1: User provides dim.
        // Option 2: Store dim in header (our binary format doesn't have one).
        // Option 3: Quick scan (still reads all pages).
        // Let's assume user provides dim or we do a quick scan if needed.
        // For ImplicitMME, we usually know n_animals from ID file.
        // Let's implement a scan method.
    }

    // Scan for max index to determine dimensions
    // This will page in the file, but sequential scan is fast.
    void scan_dim() {
        if (!mmap.is_open()) return;
        const char* ptr = (const char*)mmap.get_data();
        int max_idx = 0;

        // Use parallel scan if possible?
        // Or just single thread. Memory bandwidth is bottleneck.
        for(size_t i=0; i<n_entries; ++i) {
            const int* r_ptr = reinterpret_cast<const int*>(ptr + i*12);
            int r = r_ptr[0];
            int c = r_ptr[1];
            if (r > max_idx) max_idx = r;
            if (c > max_idx) max_idx = c;
        }
        _rows = max_idx + 1;
        _cols = max_idx + 1;
    }

    void set_dim(int r, int c) { _rows = r; _cols = c; }

    int rows() const { return _rows; }
    int cols() const { return _cols; }
    size_t nonZeros() const { return n_entries; }

    // Visit all triplets
    void visit(std::function<void(int, int, double)> callback) {
        if (!mmap.is_open()) return;
        const char* ptr = (const char*)mmap.get_data();

        // Sequential scan
        for (long long i = 0; i < (long long)n_entries; ++i) {
            const char* p = ptr + i * 12;
            int r = *reinterpret_cast<const int*>(p);
            int c = *reinterpret_cast<const int*>(p + 4);
            float v = *reinterpret_cast<const float*>(p + 8);
            callback(r, c, (double)v);
        }
    }

    // y += alpha * A * x
    void add_product_to(const Eigen::VectorXd& x, Eigen::VectorXd& y, double alpha) const {
        if (!mmap.is_open()) return;
        const char* ptr = (const char*)mmap.get_data();

        // Parallelize?
        // We can split range [0, n_entries)

        // Note: Binary format is typically unsorted or sorted by row.
        // Standard HIBLUP binary is just a list of triplets.

        // It's symmetric usually, but stored as upper/lower?
        // Usually stored as one triangle or full?
        // Our reader 'readInvMatrixBinary' auto-symmetrizes if needed.
        // MMap access implies we read AS IS.
        // If the file only contains lower triangle, we must handle symmetry manually.
        // Let's assume we handle symmetry on the fly.

        // But how do we know if it's full or symmetric?
        // Standard HIBLUP binary usually stores one triangle?
        // Let's assume we treat it as symmetric triplets: if i!=j, add both ways.
        // This is safe for inverse covariance matrices which are symmetric.

        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif

        // To avoid race conditions on y, we can use atomic or thread-local y.
        // Thread-local y is better for performance if y is small enough.
        // But y size is n_animals (potentially millions).
        // 1M doubles = 8MB. 64 threads = 512MB. Feasible.

        std::vector<Eigen::VectorXd> thread_y(num_threads);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            thread_y[tid].setZero(_rows); // Resize happens here

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (long long i = 0; i < (long long)n_entries; ++i) {
                const char* p = ptr + i * 12;
                int r = *reinterpret_cast<const int*>(p);
                int c = *reinterpret_cast<const int*>(p + 4);
                float v = *reinterpret_cast<const float*>(p + 8);

                double val = (double)v * alpha;

                if (r < _rows && c < _cols) {
                    thread_y[tid](r) += val * x(c);
                    if (r != c) {
                        thread_y[tid](c) += val * x(r);
                    }
                }
            }
        }

        // Reduce
        for(int t=0; t<num_threads; ++t) {
            y += thread_y[t];
        }
    }

    // Diagonal extraction for preconditioner
    void add_diagonal_to(Eigen::VectorXd& diag, double alpha) const {
        if (!mmap.is_open()) return;
        const char* ptr = (const char*)mmap.get_data();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < (long long)n_entries; ++i) {
            const char* p = ptr + i * 12;
            int r = *reinterpret_cast<const int*>(p);
            int c = *reinterpret_cast<const int*>(p + 4);
            float v = *reinterpret_cast<const float*>(p + 8);

            if (r == c && r < diag.size()) {
                // Potential race if multiple entries for same diagonal (unlikely for sparse unique triplets)
                // But safer to use atomic if duplicates possible.
                // Standard triplets usually unique.
                // Let's use atomic to be safe.
#ifdef _OPENMP
#pragma omp atomic
#endif
                diag(r) += (double)v * alpha;
            }
        }
    }

    // Extract a specific block [start, start+size) x [start, start+size)
    // Returns a dense matrix
    Eigen::MatrixXd get_block(int start_row, int size, double alpha) const {
        if (!mmap.is_open()) return Eigen::MatrixXd::Zero(size, size);

        Eigen::MatrixXd block = Eigen::MatrixXd::Zero(size, size);
        const char* ptr = (const char*)mmap.get_data();

        // This is a full scan of the file to find entries in the block.
        // Inefficient if done repeatedly for many blocks.
        // But for "Cluster Block Jacobi", we do it once per block.
        // If n_entries is huge (e.g. 100M) and we have 100 blocks, we scan 100 times?
        // That's 100 * 1.2GB read = 120GB read. Might be slow but acceptable for large memory systems.
        // Optimization: Single pass to fill ALL blocks?
        // But interface here requests one block.

        int end_row = start_row + size;

        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif
        std::vector<Eigen::MatrixXd> thread_blocks(num_threads, Eigen::MatrixXd::Zero(size, size));

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (long long i = 0; i < (long long)n_entries; ++i) {
                const char* p = ptr + i * 12;
                int r = *reinterpret_cast<const int*>(p);
                int c = *reinterpret_cast<const int*>(p + 4);

                if (r >= start_row && r < end_row && c >= start_row && c < end_row) {
                    float v = *reinterpret_cast<const float*>(p + 8);
                    double val = (double)v * alpha;

                    thread_blocks[tid](r - start_row, c - start_row) += val;
                    if (r != c) {
                        thread_blocks[tid](c - start_row, r - start_row) += val;
                    }
                }
            }
        }

        for(int t=0; t<num_threads; ++t) {
            block += thread_blocks[t];
        }

        return block;
    }
};

} // namespace cosmic
