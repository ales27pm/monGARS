import Foundation
import Network
import NetworkExtension
import os.log
import React
import SystemConfiguration.CaptiveNetwork

private struct CaptureSession {
  let identifier: String
  let fileURL: URL
  let manager: NETunnelProviderManager
}

@objc(DiagnosticsModule)
class DiagnosticsModule: NSObject, RCTTurboModule {
  static func moduleName() -> String! {
    "DiagnosticsModule"
  }

  static func requiresMainQueueSetup() -> Bool {
    false
  }

  private var snapshot: [String: Any] = [:]
  private var monitor: NWPathMonitor?
  private var captureSessions: [String: CaptureSession] = [:]
  private let logger = Logger(subsystem: "com.mongars.mobile", category: "Diagnostics")

  private var bundleIdentifier: String {
    "com.mongars.mobile.PacketCapture"
  }

  @objc func prepare(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.global().async {
      self.setupPathMonitor()
      self.ensureTunnelManager { result in
        switch result {
        case .success:
          resolve(nil)
        case let .failure(error):
          reject("prepare_error", error.localizedDescription, error)
        }
      }
    }
  }

  @objc func refreshNetworkSnapshot(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.global().async {
      do {
        let snapshot = try self.buildSnapshot()
        resolve(snapshot)
      } catch {
        self.logger.error("snapshot error: \(error.localizedDescription, privacy: .public)")
        reject("snapshot_error", error.localizedDescription, error)
      }
    }
  }

  @objc func listInterfaces(_ resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.global().async {
      do {
        let interfaces = try self.enumerateInterfaces()
        resolve(interfaces)
      } catch {
        reject("interfaces_error", error.localizedDescription, error)
      }
    }
  }

  @objc func startCapture(_ options: NSDictionary, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard let interfaceName = options["interfaceName"] as? String else {
      reject("capture_error", "interfaceName missing", nil)
      return
    }
    let duration = options["durationSeconds"] as? Double ?? 300
    let remoteHost = options["remoteRviHost"] as? String

    ensureTunnelManager { result in
      switch result {
      case let .success(manager):
        self.startCapture(on: manager, interface: interfaceName, duration: duration, remoteHost: remoteHost, resolve: resolve, reject: reject)
      case let .failure(error):
        reject("capture_error", error.localizedDescription, error)
      }
    }
  }

  @objc func stopCapture(_ captureId: NSString, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    guard let session = captureSessions[captureId as String] else {
      reject("stop_error", "Capture not found", nil)
      return
    }
    session.manager.connection?.stopVPNTunnel()
    captureSessions.removeValue(forKey: captureId as String)
    resolve(["captureId": session.identifier, "path": session.fileURL.path])
  }

  @objc func enablePacketTunnel(_ configuration: NSDictionary, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
    ensureTunnelManager { result in
      switch result {
      case let .success(manager):
        manager.isEnabled = true
        manager.localizedDescription = "MonGARS Diagnostics"
        let protocolConfiguration = NETunnelProviderProtocol()
        protocolConfiguration.providerBundleIdentifier = self.bundleIdentifier
        protocolConfiguration.serverAddress = configuration["serverAddress"] as? String
        manager.protocolConfiguration = protocolConfiguration
        manager.saveToPreferences { error in
          if let error {
            reject("tunnel_error", error.localizedDescription, error)
          } else {
            resolve(nil)
          }
        }
      case let .failure(error):
        reject("tunnel_error", error.localizedDescription, error)
      }
    }
  }

  private func setupPathMonitor() {
    guard monitor == nil else { return }
    let monitor = NWPathMonitor()
    monitor.pathUpdateHandler = { path in
      self.logger.debug("Path update: \(path.debugDescription, privacy: .public)")
      self.snapshot["vpnActive"] = path.usesInterfaceType(.other)
      self.snapshot["cellularActive"] = path.usesInterfaceType(.cellular)
    }
    monitor.start(queue: DispatchQueue(label: "com.mongars.mobile.path"))
    self.monitor = monitor
  }

  private func buildSnapshot() throws -> [String: Any] {
    var snapshot: [String: Any] = [:]
    snapshot["timestamp"] = ISO8601DateFormatter().string(from: Date())

    if let interfaceDetails = try? enumerateInterfaces(), let wifi = interfaceDetails.first(where: { ($0["name"] as? String)?.hasPrefix("en") == true }) {
      snapshot["interfaces"] = interfaceDetails
      snapshot["ip"] = wifi["address"] ?? ""
    } else {
      snapshot["interfaces"] = try enumerateInterfaces()
    }

    let group = DispatchGroup()
    group.enter()
    var currentNetwork: NEHotspotNetwork?
    fetchCurrentNetwork { network in
      currentNetwork = network
      group.leave()
    }
    _ = group.wait(timeout: .now() + 1)
    snapshot["ssid"] = currentNetwork?.ssid ?? NSNull()
    snapshot["bssid"] = currentNetwork?.bssid ?? NSNull()

    snapshot["vpnActive"] = self.snapshot["vpnActive"] ?? false
    snapshot["cellularActive"] = self.snapshot["cellularActive"] ?? false
    return snapshot
  }

  private func enumerateInterfaces() throws -> [[String: Any]] {
    var result: [[String: Any]] = []
    var addressList: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&addressList) == 0, let firstAddress = addressList else {
      throw NSError(domain: "Diagnostics", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unable to enumerate interfaces"])
    }
    defer { freeifaddrs(addressList) }

    var pointer = firstAddress
    while true {
      let name = String(cString: pointer.pointee.ifa_name)
      var addressString: String?
      if let addrPtr = pointer.pointee.ifa_addr, addrPtr.pointee.sa_family == UInt8(AF_INET) {
        var address = addrPtr.pointee
        var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        getnameinfo(&address, socklen_t(addrPtr.pointee.sa_len), &hostname, socklen_t(hostname.count), nil, 0, NI_NUMERICHOST)
        addressString = String(cString: hostname)
      }
      result.append([
        "name": name,
        "address": addressString ?? NSNull(),
        "mac": NSNull(),
        "isUp": (pointer.pointee.ifa_flags & UInt32(IFF_UP)) != 0,
      ])

      if let next = pointer.pointee.ifa_next {
        pointer = next
      } else {
        break
      }
    }
    return result
  }

  private func fetchCurrentNetwork(completion: @escaping (NEHotspotNetwork?) -> Void) {
    if #available(iOS 14.0, *) {
      NEHotspotNetwork.fetchCurrent { network in
        completion(network)
      }
    } else {
      completion(nil)
    }
  }

  private func ensureTunnelManager(completion: @escaping (Result<NETunnelProviderManager, Error>) -> Void) {
    NETunnelProviderManager.loadAllFromPreferences { managers, error in
      if let error {
        completion(.failure(error))
        return
      }
      if let manager = managers?.first(where: { $0.protocolConfiguration is NETunnelProviderProtocol }) {
        completion(.success(manager))
        return
      }
      let manager = NETunnelProviderManager()
      manager.protocolConfiguration = NETunnelProviderProtocol()
      manager.protocolConfiguration?.providerBundleIdentifier = self.bundleIdentifier
      manager.localizedDescription = "MonGARS Packet Capture"
      manager.isEnabled = true
      manager.saveToPreferences { error in
        if let error {
          completion(.failure(error))
        } else {
          completion(.success(manager))
        }
      }
    }
  }

  private func startCapture(on manager: NETunnelProviderManager, interface: String, duration: Double, remoteHost: String?, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
    let captureId = UUID().uuidString
    let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    let fileURL = directory.appendingPathComponent("capture-\(captureId).pcap")

    var options: [String: NSObject] = [
      "interface": interface as NSString,
      "output": fileURL.path as NSString,
      "duration": NSNumber(value: duration),
    ]
    if let remoteHost {
      options["remoteRviHost"] = remoteHost as NSString
    }

    do {
      try manager.connection?.startVPNTunnel(options: options)
    } catch {
      do {
        try manager.startVPNTunnel(options: options)
      } catch {
        reject("capture_error", error.localizedDescription, error)
        return
      }
    }

    captureSessions[captureId] = CaptureSession(identifier: captureId, fileURL: fileURL, manager: manager)
    resolve(["captureId": captureId, "path": fileURL.path])
  }
}
