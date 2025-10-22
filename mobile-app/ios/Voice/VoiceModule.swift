import AVFoundation
import Foundation
import os.log
import React
import Speech

@objc(VoiceModule)
class VoiceModule: RCTEventEmitter, RCTTurboModule {
  static func moduleName() -> String! {
    "VoiceModule"
  }

  static func requiresMainQueueSetup() -> Bool {
    true
  }

  private let logger = Logger(subsystem: "com.mongars.mobile", category: "Voice")
  private let audioEngine = AVAudioEngine()
  private let speechRecognizer = SFSpeechRecognizer()
  private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
  private var recognitionTask: SFSpeechRecognitionTask?

  override func supportedEvents() -> [String]! {
    ["onTranscript", "onTranscriptError"]
  }

  override func startObserving() {
    requestPermissions()
  }

  @objc func configureAudioSession(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    do {
      let session = AVAudioSession.sharedInstance()
      try session.setCategory(.playAndRecord, mode: .measurement, options: [.allowBluetooth, .defaultToSpeaker])
      try session.setActive(true, options: .notifyOthersOnDeactivation)
      resolve(nil)
    } catch {
      logger.error("Audio session error: \(error.localizedDescription, privacy: .public)")
      reject("audio_error", error.localizedDescription, error)
    }
  }

  @objc func startListening(_ locale: NSString?, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    requestPermissions()
    recognitionTask?.cancel()
    recognitionTask = nil

    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = true
    recognitionRequest = request

    if let localeIdentifier = locale as String? {
      speechRecognizer?.locale = Locale(identifier: localeIdentifier)
    }

    recognitionTask = speechRecognizer?.recognitionTask(with: request) { result, error in
      if let result {
        let text = result.bestTranscription.formattedString
        let payload: [String: Any] = ["text": text, "isFinal": result.isFinal]
        self.sendEvent(withName: "onTranscript", body: payload)
      }

      if let error {
        self.logger.error("Speech error: \(error.localizedDescription, privacy: .public)")
        self.sendEvent(withName: "onTranscriptError", body: ["message": error.localizedDescription])
      }
    }

    installTap()
    audioEngine.prepare()
    do {
      try audioEngine.start()
      resolve(nil)
    } catch {
      reject("voice_error", error.localizedDescription, error)
    }
  }

  @objc func stopListening(_ resolve: RCTPromiseResolveBlock, rejecter reject: RCTPromiseRejectBlock) {
    recognitionTask?.finish()
    recognitionTask = nil
    recognitionRequest?.endAudio()
    recognitionRequest = nil
    audioEngine.stop()
    audioEngine.inputNode.removeTap(onBus: 0)
    resolve(nil)
  }

  @objc func setOnResultListener() {}

  @objc func removeOnResultListener() {}

  private func requestPermissions() {
    SFSpeechRecognizer.requestAuthorization { status in
      if status != .authorized {
        self.logger.error("Speech permission denied")
      }
    }
    AVAudioSession.sharedInstance().requestRecordPermission { granted in
      if !granted {
        self.logger.error("Microphone permission denied")
      }
    }
  }

  private func installTap() {
    let inputNode = audioEngine.inputNode
    let recordingFormat = inputNode.outputFormat(forBus: 0)
    inputNode.removeTap(onBus: 0)
    inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
      self.recognitionRequest?.append(buffer)
    }
  }
}
