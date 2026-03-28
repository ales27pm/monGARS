package com.mongars.mobile

import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.VpnService
import android.net.wifi.WifiManager
import android.os.Environment
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import java.io.File
import java.net.Inet4Address
import java.util.UUID

class DiagnosticsModule(private val reactContext: ReactApplicationContext) :
  ReactContextBaseJavaModule(reactContext) {

  private val captureSessions = mutableMapOf<String, File>()

  override fun getName(): String = "DiagnosticsModule"

  @ReactMethod
  fun prepare(promise: Promise) {
    promise.resolve(null)
  }

  @ReactMethod
  fun refreshNetworkSnapshot(promise: Promise) {
    try {
      val wifiManager = reactContext.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
      val connectivityManager = reactContext.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
      val activeNetwork = connectivityManager.activeNetwork
      val capabilities = activeNetwork?.let { connectivityManager.getNetworkCapabilities(it) }
      val interfaces = listInterfacesInternal()

      val map = Arguments.createMap()
      map.putString("timestamp", java.time.Instant.now().toString())
      map.putArray("interfaces", interfaces)
      map.putBoolean("vpnActive", capabilities?.hasTransport(NetworkCapabilities.TRANSPORT_VPN) == true)
      map.putBoolean("cellularActive", capabilities?.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) == true)
      map.putString("ssid", sanitizeSsid(wifiManager.connectionInfo?.ssid))
      map.putString("ip", wifiManager.connectionInfo?.ipAddress?.let { intToIp(it) })
      promise.resolve(map)
    } catch (error: Exception) {
      promise.reject("snapshot_error", error)
    }
  }

  @ReactMethod
  fun listInterfaces(promise: Promise) {
    try {
      val array = listInterfacesInternal()
      promise.resolve(array)
    } catch (error: Exception) {
      promise.reject("interfaces_error", error)
    }
  }

  @ReactMethod
  fun startCapture(options: ReadableMap, promise: Promise) {
    val interfaceName = options.getString("interfaceName") ?: run {
      promise.reject("capture_error", IllegalArgumentException("interfaceName missing"))
      return
    }
    if (currentActivity == null) {
      promise.reject("capture_error", IllegalStateException("Activity required"))
      return
    }
    val durationSeconds = if (options.hasKey("durationSeconds")) options.getDouble("durationSeconds") else 300.0
    val captureId = UUID.randomUUID().toString()
    val outputDir = reactContext.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS) ?: reactContext.filesDir
    val output = File(outputDir, "capture-$captureId.pcap")
    captureSessions[captureId] = output

    val intent = VpnService.prepare(reactContext)
    if (intent != null) {
      captureSessions.remove(captureId)
      promise.reject(
        "capture_error",
        "VPN permission required. Call prepare() and obtain consent before startCapture.",
        null,
      )
      return
    }

    val serviceIntent = Intent(reactContext, PacketCaptureService::class.java)
    serviceIntent.putExtra(PacketCaptureService.EXTRA_INTERFACE, interfaceName)
    serviceIntent.putExtra(PacketCaptureService.EXTRA_DURATION, durationSeconds.toLong())
    serviceIntent.putExtra(PacketCaptureService.EXTRA_OUTPUT, output.absolutePath)
    reactContext.startService(serviceIntent)

    val result = Arguments.createMap()
    result.putString("captureId", captureId)
    result.putString("path", output.absolutePath)
    promise.resolve(result)
  }

  @ReactMethod
  fun stopCapture(captureId: String, promise: Promise) {
    val output = captureSessions[captureId]
      ?: run {
        promise.reject("stop_error", IllegalArgumentException("Capture not found"))
        return
      }
    val serviceIntent = Intent(reactContext, PacketCaptureService::class.java)
    serviceIntent.action = PacketCaptureService.ACTION_STOP
    reactContext.startService(serviceIntent)
    captureSessions.remove(captureId)

    val result = Arguments.createMap()
    result.putString("captureId", captureId)
    result.putString("path", output?.absolutePath)
    promise.resolve(result)
  }

  @ReactMethod
  fun enablePacketTunnel(configuration: ReadableMap, promise: Promise) {
    // Android leverages the VpnService-based capture.
    promise.resolve(null)
  }

  private fun listInterfacesInternal(): com.facebook.react.bridge.WritableArray {
    val array = Arguments.createArray()
    val interfaces = java.net.NetworkInterface.getNetworkInterfaces()
    while (interfaces.hasMoreElements()) {
      val networkInterface = interfaces.nextElement()
      val item = Arguments.createMap()
      item.putString("name", networkInterface.name)
      val addresses = networkInterface.inetAddresses.toList().filterIsInstance<Inet4Address>()
      item.putString("address", addresses.firstOrNull()?.hostAddress)
      item.putString("mac", networkInterface.hardwareAddress?.joinToString(":") { b -> String.format("%02X", b) })
      item.putBoolean("isUp", networkInterface.isUp)
      array.pushMap(item)
    }
    return array
  }

  private fun intToIp(ip: Int): String {
    return ((ip and 0xFF).toString() + "." + (ip shr 8 and 0xFF) + "." + (ip shr 16 and 0xFF) + "." + (ip shr 24 and 0xFF))
  }

  private fun sanitizeSsid(value: String?): String? {
    if (value.isNullOrBlank() || value == "<unknown ssid>") {
      return null
    }
    return value.trim('"')
  }
}
