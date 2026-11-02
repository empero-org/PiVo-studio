// iOS Swift wrapper for pivo_runner — uses a bridging header that imports
// `include/pivo_ffi.h`. See docs/MOBILE.md for the Xcode build recipe.
//
// Add to your <App>-Bridging-Header.h:
//   #import "pivo_ffi.h"
//
// Then:
//   let engine = Pivo.load(bundleURL: url)
//   let wav = try engine.synthesizeWav("NASA is ready.", voice: "Pi", speed: 1.0)
//   for await frame in try engine.stream("Long text…") { /* play 160 samples */ }

import Foundation

public enum PivoError: Error, CustomStringConvertible {
    case loadFailed(String)
    case synthesizeFailed(String)
    case streamFailed(String)

    public var description: String {
        switch self {
        case .loadFailed(let m): return "pivo load failed: \(m)"
        case .synthesizeFailed(let m): return "pivo synthesize failed: \(m)"
        case .streamFailed(let m): return "pivo stream failed: \(m)"
        }
    }
}

public final class PivoEngine {
    private let handle: OpaquePointer

    public init(directory: URL) throws {
        var out: OpaquePointer?
        directory.path.withCString { path in
            out = pivo_load_model(path)
        }
        guard let h = out else {
            throw PivoError.loadFailed(lastError() ?? "unknown error")
        }
        handle = h
    }

    public init(bundleData: Data) throws {
        var out: OpaquePointer?
        bundleData.withUnsafeBytes { buf in
            out = pivo_load_bundle(buf.bindMemory(to: UInt8.self).baseAddress, buf.count)
        }
        guard let h = out else {
            throw PivoError.loadFailed(lastError() ?? "unknown error")
        }
        handle = h
    }

    deinit { pivo_free_engine(handle) }

    public var sampleRate: Int { Int(pivo_sample_rate(handle)) }

    /// Samples per output frame (hop_length; 160 in v1 — always query this).
    public var frameSamples: Int { Int(pivo_frame_samples(handle)) }

    private func makeOptions(voice: String, speed: Float, strict: Bool,
                             overridesJson: String? = nil,
                             then: (UnsafePointer<PivoOptions>) -> Void) {
        voice.withCString { v in
            (overridesJson ?? "").withCString { oj in
                var opts = PivoOptions(
                    voice: v, speed: speed, strict: strict ? 1 : 0,
                    overrides_json: overridesJson == nil ? nil : oj)
                withUnsafePointer(to: &opts) { then($0) }
            }
        }
    }

    public func synthesizeWav(_ text: String, voice: String = "Pi",
                            speed: Float = 1.0, strict: Bool = true,
                            overridesJson: String? = nil) throws -> Data {
        var len: Int = 0
        var result: UnsafeMutablePointer<UInt8>?
        text.withCString { t in
            makeOptions(voice: voice, speed: speed, strict: strict,
                        overridesJson: overridesJson) { opts in
                result = pivo_synthesize_wav(handle, t, opts, &len)
            }
        }
        guard let wav = result else {
            throw PivoError.synthesizeFailed(lastError() ?? "unknown error")
        }
        defer { pivo_free_buffer(wav, len) }
        return Data(bytes: wav, count: len)
    }

    /// Stream 10 ms f32 frames as an AsyncSequence (Swift 5.7+).
    public func stream(_ text: String, voice: String = "Pi",
                       speed: Float = 1.0, strict: Bool = false)
        -> AsyncThrowingStream<[Float], Error> {
        AsyncThrowingStream { continuation in
            Task.detached { [handle] in
                var streamPtr: OpaquePointer?
                text.withCString { t in
                    self.makeOptions(voice: voice, speed: speed, strict: strict) { opts in
                        streamPtr = pivo_stream_new(handle, t, opts)
                    }
                }
                guard let s = streamPtr else {
                    continuation.finish(throwing: PivoError.streamFailed(lastError() ?? "unknown error"))
                    return
                }
                defer { pivo_stream_free(s) }
                var frame = [Float](repeating: 0, count: self.frameSamples)
                while true {
                    let n = frame.withUnsafeMutableBufferPointer {
                        pivo_stream_next_frame(s, $0.baseAddress, self.frameSamples)
                    }
                    if n > 0 {
                        continuation.yield(Array(frame.prefix(n)))
                    } else if n == 0 {
                        continuation.finish()
                        return
                    } else {
                        continuation.finish(throwing:
                            PivoError.streamFailed(lastError() ?? "read error"))
                        return
                    }
                }
            }
        }
    }
}

private func lastError() -> String? {
    guard let p = pivo_last_error_message() else { return nil }
    return String(cString: p)
}
