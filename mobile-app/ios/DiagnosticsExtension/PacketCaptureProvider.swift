import NetworkExtension
import os.log

class PacketCaptureProvider: NEPacketTunnelProvider {
  private let logger = Logger(subsystem: "com.mongars.mobile", category: "PacketCapture")
  private var fileHandle: FileHandle?
  private var stopWorkItem: DispatchWorkItem?
  private var captureWorkItem: DispatchWorkItem?
  private var isCapturing = false

  override func startTunnel(options: [String: NSObject]?, completionHandler: @escaping (Error?) -> Void) {
    logger.log("Starting packet tunnel with options: \(String(describing: options), privacy: .public)")

    let settings = NEPacketTunnelNetworkSettings(tunnelRemoteAddress: "127.0.0.1")
    settings.ipv4Settings = NEIPv4Settings(addresses: ["192.0.2.1"], subnetMasks: ["255.255.255.0"])
    settings.ipv4Settings?.includedRoutes = [NEIPv4Route.default()]
    settings.mtu = 1500

    setTunnelNetworkSettings(settings) { error in
      if let error {
        completionHandler(error)
        return
      }
      self.beginCapture(options: options)
      completionHandler(nil)
    }
  }

  override func stopTunnel(with reason: NEProviderStopReason, completionHandler: @escaping () -> Void) {
    logger.log("Stopping packet tunnel: \(reason.rawValue)")
    isCapturing = false
    captureWorkItem?.cancel()
    captureWorkItem = nil
    stopWorkItem?.cancel()
    stopWorkItem = nil
    do {
      try fileHandle?.close()
    } catch {
      logger.error("Unable to close file: \(error.localizedDescription, privacy: .public)")
    }
    fileHandle = nil
    completionHandler()
  }

  private func beginCapture(options: [String: NSObject]?) {
    let outputPath = options?["output"] as? String ?? (FileManager.default.temporaryDirectory.appendingPathComponent("capture.pcap").path)
    FileManager.default.createFile(atPath: outputPath, contents: nil, attributes: nil)

    do {
      fileHandle = try FileHandle(forWritingTo: URL(fileURLWithPath: outputPath))
      writePcapHeader()
    } catch {
      logger.error("Unable to open capture file: \(error.localizedDescription, privacy: .public)")
      return
    }

    let duration = options?["duration"] as? Double ?? 300
    isCapturing = true
    let workItem = DispatchWorkItem { [weak self] in
      guard let self else { return }
      func loopRead() {
        self.packetFlow.readPackets { packets, protocols in
          guard self.isCapturing else { return }
          for (index, packet) in packets.enumerated() {
            self.writePacket(packet, protocolFamily: protocols[index])
          }
          if self.isCapturing {
            loopRead()
          }
        }
      }
      loopRead()
    }
    captureWorkItem = workItem
    stopWorkItem = DispatchWorkItem { [weak self] in
      self?.cancelTunnelWithError(nil)
    }
    DispatchQueue.global(qos: .userInitiated).async(execute: workItem)
    if let stopWorkItem {
      let deadline = DispatchTime.now() + .milliseconds(Int(duration * 1000))
      DispatchQueue.global().asyncAfter(deadline: deadline, execute: stopWorkItem)
    }
  }

  private func writePcapHeader() {
    guard let fileHandle else { return }
    var header = PcapFileHeader(
      magic: 0xa1b2c3d4,
      version_major: 2,
      version_minor: 4,
      thiszone: 0,
      sigfigs: 0,
      snaplen: 65535,
      linktype: 101 // LINKTYPE_RAW
    )
    let data = withUnsafeBytes(of: &header) { Data($0) }
    fileHandle.write(data)
  }

  private func writePacket(_ packet: Data, protocolFamily: NSNumber) {
    guard let fileHandle else { return }
    let now = Date().timeIntervalSince1970
    let seconds = UInt32(now)
    let microseconds = UInt32((now - Double(seconds)) * 1_000_000)
    var header = PcapPacketHeader(
      ts_sec: seconds,
      ts_usec: microseconds,
      caplen: UInt32(packet.count),
      len: UInt32(packet.count)
    )
    let headerData = withUnsafeBytes(of: &header) { Data($0) }
    fileHandle.write(headerData)
    fileHandle.write(packet)
  }
}

struct PcapFileHeader {
  var magic: UInt32
  var version_major: UInt16
  var version_minor: UInt16
  var thiszone: Int32
  var sigfigs: UInt32
  var snaplen: UInt32
  var linktype: UInt32
}

struct PcapPacketHeader {
  var ts_sec: UInt32
  var ts_usec: UInt32
  var caplen: UInt32
  var len: UInt32
}
