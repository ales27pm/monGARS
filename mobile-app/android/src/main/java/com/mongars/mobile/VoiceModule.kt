package com.mongars.mobile

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.DeviceEventManagerModule

class VoiceModule(private val context: ReactApplicationContext) :
  ReactContextBaseJavaModule(context), ActivityCompat.OnRequestPermissionsResultCallback {

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
    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
    intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, locale ?: "fr-FR")
    intent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
    listening = true
    speechRecognizer?.startListening(intent)
    promise.resolve(null)
  }

  @ReactMethod
  fun stopListening(promise: Promise) {
    speechRecognizer?.stopListening()
    listening = false
    promise.resolve(null)
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
}
