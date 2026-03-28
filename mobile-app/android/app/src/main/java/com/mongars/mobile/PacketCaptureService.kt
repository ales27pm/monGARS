package com.mongars.mobile

import android.content.Intent
import android.net.VpnService
import android.os.ParcelFileDescriptor
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.jvm.Volatile

class PacketCaptureService : VpnService() {
  companion object {
    const val EXTRA_INTERFACE = "interface"
    const val EXTRA_OUTPUT = "output"
    const val EXTRA_DURATION = "duration"
    const val REQUEST_CODE = 4242
    const val ACTION_STOP = "com.mongars.mobile.STOP_CAPTURE"
  }

  private val executor = Executors.newSingleThreadExecutor()
  private var vpnInterface: ParcelFileDescriptor? = null
  private var captureFile: File? = null
  @Volatile private var shouldRun = true

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    if (intent?.action == ACTION_STOP) {
      stopCapture()
      return START_NOT_STICKY
    }
    val interfaceName = intent?.getStringExtra(EXTRA_INTERFACE) ?: "tun0"
    val outputPath = intent?.getStringExtra(EXTRA_OUTPUT) ?: return START_NOT_STICKY
    val duration = intent.getLongExtra(EXTRA_DURATION, 300)

    captureFile = File(outputPath)
    shouldRun = true
    executor.execute { capture(interfaceName, duration) }
    return START_STICKY
  }

  private fun capture(interfaceName: String, duration: Long) {
    try {
      val builder = Builder()
      builder.setSession("MonGARS Capture")
      builder.addAddress("10.0.0.2", 32)
      builder.addRoute("0.0.0.0", 0)
      vpnInterface = builder.establish()
      val pfd = vpnInterface ?: return
      FileOutputStream(captureFile).use { stream ->
        writePcapHeader(stream)
        val packetHeaderBuffer = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
        val buffer = ByteArray(32768)
        ParcelFileDescriptor.AutoCloseInputStream(pfd).use { input ->
          val start = System.currentTimeMillis()
          while (shouldRun && System.currentTimeMillis() - start < duration * 1000) {
            val read = input.read(buffer)
            if (read > 0) {
              writePacket(stream, packetHeaderBuffer, buffer, read)
            }
          }
        }
      }
    } catch (error: Exception) {
      Log.e("PacketCapture", "Capture failed", error)
    } finally {
      stopCapture()
    }
  }

  private fun stopCapture() {
    shouldRun = false
    vpnInterface?.close()
    vpnInterface = null
    stopSelf()
  }

  override fun onDestroy() {
    super.onDestroy()
    stopCapture()
  }

  private fun writePcapHeader(stream: FileOutputStream) {
    val headerBuffer = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN)
    headerBuffer.putInt(0xa1b2c3d4L.toInt())
    headerBuffer.putShort(2.toShort())
    headerBuffer.putShort(4.toShort())
    headerBuffer.putInt(0)
    headerBuffer.putInt(0)
    headerBuffer.putInt(65535)
    headerBuffer.putInt(101)
    stream.write(headerBuffer.array())
  }

  private fun writePacket(
    stream: FileOutputStream,
    packetHeaderBuffer: ByteBuffer,
    packet: ByteArray,
    length: Int,
  ) {
    val nowMillis = System.currentTimeMillis()
    val seconds = (nowMillis / 1000L).toInt()
    val microseconds = ((nowMillis % 1000L) * 1000L).toInt()

    packetHeaderBuffer.clear()
    packetHeaderBuffer.putInt(seconds)
    packetHeaderBuffer.putInt(microseconds)
    packetHeaderBuffer.putInt(length)
    packetHeaderBuffer.putInt(length)

    stream.write(packetHeaderBuffer.array(), 0, packetHeaderBuffer.position())
    stream.write(packet, 0, length)
  }
}

