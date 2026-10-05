#pragma once

#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Disk-backed mappings use reclaimable page cache, not a second anonymous KV copy.
namespace server_sleep_cache {

inline std::string key() {
    const char * value = std::getenv("LLAMA_SLEEP_CACHE_KEY");
    if (!value || !*value) {
        throw std::runtime_error("LLAMA_SLEEP_CACHE_KEY is required");
    }
    return value;
}

inline size_t limit() {
    const char * value = std::getenv("LLAMA_SLEEP_CACHE_MAX_MIB");
    char * end = nullptr;
    const unsigned long long mib = value ? std::strtoull(value, &end, 10) : 16384;
    if (mib == 0 || (value && (!*value || *end)) || mib > SIZE_MAX / (1024 * 1024)) {
        throw std::runtime_error("invalid LLAMA_SLEEP_CACHE_MAX_MIB");
    }
    return size_t(mib) * 1024 * 1024;
}

class mapping {
    int fd = -1;
    std::string temporary;
public:
    uint8_t * data = nullptr;
    size_t size = 0;

    mapping(const std::string & path, size_t write_size = 0) {
        try {
            if (write_size) {
                std::string pattern = path + ".tmp.XXXXXX";
                std::vector<char> name(pattern.begin(), pattern.end());
                name.push_back('\0');
                fd = mkstemp(name.data());
                if (fd < 0) {
                    throw std::runtime_error(std::strerror(errno));
                }
                temporary = name.data();
                // Reserve real disk space before mmap writes, avoiding ENOSPC SIGBUS.
                const int error = posix_fallocate(fd, 0, write_size);
                if (error) {
                    throw std::runtime_error(std::strerror(error));
                }
                size = write_size;
            } else {
                fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
                if (fd < 0) {
                    throw std::runtime_error(std::strerror(errno));
                }
                struct stat st;
                if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0 || uint64_t(st.st_size) > limit()) {
                    throw std::runtime_error("invalid snapshot file size or type");
                }
                size = st.st_size;
            }
            void * ptr = mmap(nullptr, size, PROT_READ | (write_size ? PROT_WRITE : 0), MAP_SHARED, fd, 0);
            if (ptr == MAP_FAILED) {
                throw std::runtime_error(std::strerror(errno));
            }
            data = static_cast<uint8_t *>(ptr);
        } catch (...) {
            close_file();
            throw;
        }
    }

    mapping(const mapping &) = delete;
    mapping & operator=(const mapping &) = delete;
    ~mapping() { close_file(); }

    void close_file() {
        if (data) {
            munmap(data, size);
            data = nullptr;
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
        if (!temporary.empty()) {
            unlink(temporary.c_str());
            temporary.clear();
        }
    }

    void commit(const std::string & path) {
        if (msync(data, size, MS_SYNC) || fsync(fd) || rename(temporary.c_str(), path.c_str())) {
            throw std::runtime_error(std::strerror(errno));
        }
        temporary.clear();
    }
};

struct cursor {
    uint8_t * data;
    size_t size;
    size_t pos = 0;

    uint8_t * take(size_t count) {
        if (count > size - pos) {
            throw std::runtime_error("snapshot is truncated or exceeds its size limit");
        }
        uint8_t * result = data ? data + pos : nullptr;
        pos += count;
        return result;
    }

    template<typename T> void write(T value) {
        uint8_t * ptr = take(sizeof(value));
        if (ptr) {
            std::memcpy(ptr, &value, sizeof(value));
        }
    }

    template<typename T> T read() {
        T value;
        std::memcpy(&value, take(sizeof(value)), sizeof(value));
        return value;
    }

    void write_blob(const void * source, size_t count) {
        write<uint64_t>(count);
        uint8_t * ptr = take(count);
        if (ptr && count) {
            std::memcpy(ptr, source, count);
        }
    }

    std::pair<const uint8_t *, size_t> read_blob() {
        const uint64_t count = read<uint64_t>();
        if (count > SIZE_MAX) {
            throw std::runtime_error("snapshot blob is too large");
        }
        return {take(count), size_t(count)};
    }

    template<typename T> std::vector<T> read_vector() {
        const auto blob = read_blob();
        if (blob.second % sizeof(T)) {
            throw std::runtime_error("invalid snapshot vector size");
        }
        std::vector<T> result(blob.second / sizeof(T));
        if (blob.second) {
            std::memcpy(result.data(), blob.first, blob.second);
        }
        return result;
    }
};

constexpr uint64_t magic = 0x3145484341434c4c; // LLCACHE1

inline uint64_t checksum(const uint8_t * data, size_t size) {
    uint64_t result = 0xcbf29ce484222325;
    while (size >= sizeof(uint64_t)) {
        uint64_t word;
        std::memcpy(&word, data, sizeof(word));
        result = (result ^ word) * 0x100000001b3;
        data += sizeof(word);
        size -= sizeof(word);
    }
    while (size--) {
        result = (result ^ *data++) * 0x100000001b3;
    }
    return result;
}

template<typename Slot> size_t save(const std::string & path, const Slot & slot) {
    if (slot.prompt.tokens.size() == 0) {
        std::error_code error;
        std::filesystem::remove(path, error);
        return 0;
    }
    const auto identity = key();
    const auto tokens = slot.prompt.tokens.serialize();
    std::vector<uint8_t> spec_state;
    common_speculative_get_state(slot.spec, slot.id, spec_state);
    const size_t main_size = llama_state_seq_get_size_ext(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
    const size_t draft_size = slot.ctx_dft ? llama_state_seq_get_size_ext(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    if (!main_size || (slot.ctx_dft && !draft_size)) {
        throw std::runtime_error("cannot measure sequence state");
    }

    auto metadata = [&](cursor & out) {
        out.write<uint64_t>(magic);
        out.write<int32_t>(slot.id);
        out.write<int32_t>(slot.n_ctx);
        out.write_blob(identity.data(), identity.size());
        out.write_blob(tokens.data(), tokens.size());
        out.write_blob(spec_state.data(), spec_state.size());
        out.write<uint64_t>(slot.prompt.checkpoints.size());
        for (const auto & checkpoint : slot.prompt.checkpoints) {
            out.write<int64_t>(checkpoint.n_tokens);
            out.write<llama_pos>(checkpoint.pos_min);
            out.write<llama_pos>(checkpoint.pos_max);
            out.write_blob(checkpoint.data_tgt.data(), checkpoint.data_tgt.size());
            out.write_blob(checkpoint.data_dft.data(), checkpoint.data_dft.size());
            out.write_blob(checkpoint.data_spec.data(), checkpoint.data_spec.size());
        }
    };

    cursor measure{nullptr, limit()};
    metadata(measure);
    measure.write<uint64_t>(main_size);
    measure.take(main_size);
    measure.write<uint64_t>(draft_size);
    measure.take(draft_size);
    measure.write<uint64_t>(0); // Whole-file corruption checksum, excluding this final word.
    mapping file(path, measure.pos);
    cursor out{file.data, file.size};
    metadata(out);
    auto save_context = [&](llama_context * ctx, size_t count) {
        out.write<uint64_t>(count);
        uint8_t * ptr = out.take(count);
        if (count && llama_state_seq_get_data_ext(ctx, ptr, count, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != count) {
            throw std::runtime_error("cannot save sequence state");
        }
    };
    save_context(slot.ctx_tgt, main_size);
    save_context(slot.ctx_dft, draft_size);
    const uint64_t digest = checksum(file.data, out.pos);
    out.write<uint64_t>(digest);
    file.commit(path);
    return file.size;
}

template<typename Slot> size_t restore(const std::string & path, Slot & slot) {
    if (!std::filesystem::exists(path)) {
        return 0;
    }
    mapping file(path);
    if (file.size < sizeof(uint64_t)) {
        throw std::runtime_error("snapshot is truncated");
    }
    cursor in{file.data, file.size - sizeof(uint64_t)};
    if (in.read<uint64_t>() != magic || in.read<int32_t>() != slot.id || in.read<int32_t>() != slot.n_ctx) {
        throw std::runtime_error("incompatible snapshot header");
    }
    const auto identity = in.read_blob();
    if (identity.second > 1024 || std::string(reinterpret_cast<const char *>(identity.first), identity.second) != key()) {
        throw std::runtime_error("snapshot belongs to a different model or configuration");
    }
    uint64_t digest;
    std::memcpy(&digest, file.data + in.size, sizeof(digest));
    if (checksum(file.data, in.size) != digest) {
        throw std::runtime_error("snapshot checksum mismatch");
    }
    server_prompt prompt;
    prompt.tokens = server_tokens::deserialize(in.read_vector<llama_token>(), slot.mctx != nullptr);
    if (prompt.tokens.size() > size_t(slot.n_ctx) || !prompt.tokens.validate(slot.ctx_tgt)) {
        throw std::runtime_error("invalid snapshot tokens");
    }
    const auto spec_state = in.read_vector<uint8_t>();
    const uint64_t count = in.read<uint64_t>();
    if (count > 32) {
        throw std::runtime_error("too many snapshot checkpoints");
    }
    for (uint64_t i = 0; i < count; ++i) {
        auto & checkpoint = prompt.checkpoints.emplace_back();
        checkpoint.n_tokens = in.read<int64_t>();
        checkpoint.pos_min = in.read<llama_pos>();
        checkpoint.pos_max = in.read<llama_pos>();
        if (checkpoint.n_tokens < 0 || checkpoint.n_tokens > int64_t(prompt.tokens.size()) ||
                checkpoint.pos_min < 0 || checkpoint.pos_max < checkpoint.pos_min || checkpoint.pos_max >= slot.n_ctx) {
            throw std::runtime_error("invalid snapshot checkpoint positions");
        }
        checkpoint.data_tgt = in.read_vector<uint8_t>();
        checkpoint.data_dft = in.read_vector<uint8_t>();
        checkpoint.data_spec = in.read_vector<uint8_t>();
    }
    const auto main_state = in.read_blob();
    const auto draft_state = in.read_blob();
    if (in.pos != in.size || !main_state.second || bool(draft_state.second) != bool(slot.ctx_dft)) {
        throw std::runtime_error("invalid snapshot sequence state");
    }
    if (llama_state_seq_set_data_ext(slot.ctx_tgt, main_state.first, main_state.second, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != main_state.second ||
            (slot.ctx_dft && llama_state_seq_set_data_ext(slot.ctx_dft, draft_state.first, draft_state.second, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != draft_state.second)) {
        throw std::runtime_error("cannot restore sequence state");
    }
    common_speculative_set_state(slot.spec, slot.id, spec_state);
    slot.prompt = std::move(prompt);
    return file.size;
}

} // namespace server_sleep_cache
