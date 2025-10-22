package com.mongars.mobile

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.UiThreadUtil
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.util.Locale

class VoiceModule(private val context: ReactApplicationContext) :
  ReactContextBaseJavaModule(context) {

  private var speechRecognizer: SpeechRecognizer? = null
  private var listening = false

  override fun getName(): String = "VoiceModule"

  @ReactMethod
  fun configureAudioSession(promise: Promise) {
    requestPermissions()
    promise.resolve(null)
  }

  @ReactMethod
  fun startListening(locale: String?, promise: Promise) {
    requestPermissions()
    if (!hasRecordAudioPermission()) {
      promise.reject("E_PERMISSION", "RECORD_AUDIO permission not granted")
      return
    }
    if (!SpeechRecognizer.isRecognitionAvailable(context)) {
      promise.reject("E_UNAVAILABLE", "Speech recognition not available on device")
      return
    }

    UiThreadUtil.runOnUiThread {
      if (speechRecognizer == null) {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
          override fun onReadyForSpeech(params: Bundle?) {}
          override fun onBeginningOfSpeech() {}
          override fun onRmsChanged(rmsdB: Float) {}
          override fun onBufferReceived(buffer: ByteArray?) {}
          override fun onEndOfSpeech() {}
          override fun onError(error: Int) {
            sendError("Speech error: $error")
            listening = false
          }
          override fun onResults(results: Bundle) {
            val matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            matches?.firstOrNull()?.let { sendTranscript(it, true) }
            listening = false
          }
          override fun onPartialResults(partialResults: Bundle) {
            val matches = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            matches?.firstOrNull()?.let { sendTranscript(it, false) }
          }
          override fun onEvent(eventType: Int, params: Bundle?) {}
        })
      }

      val language = locale ?: Locale.getDefault().toLanguageTag()
      val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, language)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
      }
      listening = true
      speechRecognizer?.startListening(intent)
      promise.resolve(null)
    }
  }

  @ReactMethod
  fun stopListening(promise: Promise) {
    UiThreadUtil.runOnUiThread {
      speechRecognizer?.stopListening()
      listening = false
      promise.resolve(null)
    }
  }

  @ReactMethod
  fun setOnResultListener() {}

  @ReactMethod
  fun removeOnResultListener() {}

  private fun sendTranscript(text: String, isFinal: Boolean) {
    val payload = Arguments.createMap()
    payload.putString("text", text)
    payload.putBoolean("isFinal", isFinal)
    context.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
      .emit("onTranscript", payload)
  }

  private fun sendError(message: String) {
    val payload = Arguments.createMap()
    payload.putString("message", message)
    context.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
      .emit("onTranscriptError", payload)
  }

  private fun requestPermissions() {
    val micPermission = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
    if (micPermission != PackageManager.PERMISSION_GRANTED && currentActivity != null) {
      ActivityCompat.requestPermissions(currentActivity!!, arrayOf(Manifest.permission.RECORD_AUDIO), 9921)
    }
  }

  private fun hasRecordAudioPermission(): Boolean {
    return ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
  }

  override fun onCatalystInstanceDestroy() {
    UiThreadUtil.runOnUiThread {
      speechRecognizer?.destroy()
      speechRecognizer = null
      listening = false
    }
    super.onCatalystInstanceDestroy()
  }
}
