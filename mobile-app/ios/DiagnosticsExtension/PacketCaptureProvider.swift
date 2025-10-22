import NetworkExtension
import os.log

class PacketCaptureProvider: NEPacketTunnelProvider {
  private let logger = Logger(subsystem: "com.mongars.mobile", category: "PacketCapture")
  private var fileHandle: FileHandle?
  private var stopWorkItem: DispatchWorkItem?

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
    }

    let duration = options?["duration"] as? Double ?? 300
    let workItem = DispatchWorkItem { [weak self] in
      self?.packetFlow.readPackets { packets, protocols in
        guard let self else { return }
        for (index, packet) in packets.enumerated() {
          self.writePacket(packet, protocolFamily: protocols[index])
        }
        self.packetFlow.readPackets { packets, protocols in
          guard let self else { return }
          for (index, packet) in packets.enumerated() {
            self.writePacket(packet, protocolFamily: protocols[index])
          }
        }
      }
    }
    stopWorkItem = DispatchWorkItem { [weak self] in
      self?.cancelTunnelWithError(nil)
    }
    DispatchQueue.global(qos: .userInitiated).async(execute: workItem)
    DispatchQueue.global().asyncAfter(deadline: .now() + duration, execute: stopWorkItem!)
  }

  private func writePcapHeader() {
    guard let fileHandle else { return }
    var header = pcap_file_header(
      magic: 0xa1b2c3d4,
      version_major: 2,
      version_minor: 4,
      thiszone: 0,
      sigfigs: 0,
      snaplen: 65535,
      linktype: 1
    )
    let data = Data(bytes: &header, count: MemoryLayout<pcap_file_header>.size)
    fileHandle.write(data)
  }

  private func writePacket(_ packet: Data, protocolFamily: NSNumber) {
    guard let fileHandle else { return }
    var header = pcap_sf_pkthdr(
      ts: timeval(tv_sec: time(nil), tv_usec: 0),
      caplen: UInt32(packet.count),
      len: UInt32(packet.count)
    )
    let headerData = Data(bytes: &header, count: MemoryLayout<pcap_sf_pkthdr>.size)
    fileHandle.write(headerData)
    fileHandle.write(packet)
  }
}

struct pcap_file_header {
  var magic: UInt32
  var version_major: UInt16
  var version_minor: UInt16
  var thiszone: Int32
  var sigfigs: UInt32
  var snaplen: UInt32
  var linktype: UInt32
}

struct pcap_sf_pkthdr {
  var ts: timeval
  var caplen: UInt32
  var len: UInt32
}
