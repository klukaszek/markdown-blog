---
title: "Apple Sensor Kit"
author: "Kyle Lukaszek"
date: "November 6th 2024"
description: "A guide to implementing ambient light sensing with Apple's SensorKit."
tags: # This is a YAML list/array
  - Swift
  - iOS
  - SensorKit
  - Research
---

# Using Apple's SensorKit for Ambient Light Monitoring on iOS Devices

This guide explains how to implement ambient light sensing in iOS applications using Apple's SensorKit framework. The implementation is based on a practical example that collects and manages ambient light sensor data.

I will not be touching on any of the UI components of the attached demo application. This is all just regular Swift iOS development and several resources are available to learn how to begin building apps with UIKit. This short article only outlines a simple process for collecting ambient light samples, but could be extended to the other sensors present on a SensorKit compatible device.

This article also does not discuss setting up SensorKit permissions with individual Apple Developer IDs. You must contact whoever is in charge of this within your organization and have them grant you special permissions with certain provisions (after Apple responds). Once this has been done, you should be able to use automatic signing and XCode should not complain.

[!NOTE]
> Devices that do not have the `SensorKit` dylib installed will immediately crash due to loader issues. So far I have not found any way to get SensorKit working on the iPad Pro even though it supposedly has the sensors we need. Works great on my iPhone 13 Pro however!

## Prerequisites

- Access to SensorKit! If you are reading this and you're one of Denis' students, lucky you, you've got access! 
- Otherwise you can apply here -> [Apple Research and Care](https://www.researchandcare.org/resources/accessing-sensorkit-data/)
    - The process can be quite lengthy and might even take a few months but I promise they will eventually get back to you (eventually...).

- Import required frameworks:
```swift
import SensorKit
import UIKit // Used in the app but not actually needed for SensorKit
```

## Data Structure

The implementation uses a custom structure that extends Apple's `SRAmbientLightSample` functionality by adding timestamp information:

```swift
struct SVILightSample {
    var lux: Measurement<UnitIlluminance>
    var chromacity: SRAmbientLightSample.Chromaticity
    var placement: SRAmbientLightSample.SensorPlacement
    var date: CFAbsoluteTime
}
```

This structure mirrors the properties of `SRAmbientLightSample` but adds a `date` field using `CFAbsoluteTime`. This addition enables easier temporal filtering and sorting of samples. When converting from `SRAmbientLightSample` to our custom type, we preserve all the original sensor data while adding the timestamp.

The added timestamp enables operations like:
- Chronological sorting of samples
- Filtering samples by time period
- Tracking data collection intervals

## Implementation Steps

### 1. Create a Manager Class

Create a class that implements `SRSensorReaderDelegate` to handle sensor data:

```swift
class LightManager: NSObject, ObservableObject, SRSensorReaderDelegate {
    private var samples: [SVILightSample] = []
    private var sensorReader = SRSensorReader(sensor: .ambientLightSensor)
    private var fetchRequest = SRFetchRequest()
    private var device = SRDevice() // This can be an array if necessary however in this article we are only working with 1 sensor.

    override init() {
        super.init()
        setupSensorKit()
    }

    func setupSensorKit() {
        print("Setting up SensorKit...")
        sensorReader.delegate = self
        requestAuthorization() // -> continued in the next section
    }


    // ...
}
```

You will notice that our Manager class also extends some other classes:
- NSObject
    - The root class for Obj-C classes (misc object functionality)
- ObservableObject
    - The ObservableObject class is extended to implement our manager in Swift UI elements.

Here we also override the `init` function and sneak in our own little setup function. Of course, I could have just put the code from that function into the init() method, but this better outlines the steps to be taken in my opinion.

In `setupSensorKit()` we assign the delegate class/object of our `SensorReader` to be our `LightManager` class. This means that our `SensorReader` is expecting the `LightManager` class to implement the required delegate functions to ensure proper functionality of SensorKit. We will touch on a few of those delegate methods later in this article.

### 2. Request Authorization

SensorKit requires explicit user authorization. Continuing from the previous code sampleImplement the authorization request:

```swift
private func requestAuthorization() {
    if (sensorReader.authorizationStatus != SRAuthorizationStatus.authorized) {
        SRSensorReader.requestAuthorization(sensors: [.ambientLightSensor]) { error in
            // Handle authorization errors
            let srError = error as? SRError
            let nsError = srError as? NSError
            
            // Annoying error handling to make sure the error is non-fatal. (nested errors :D)
            if (nsError?.code as? Int == 4) {
                let underlyingError = nsError?.underlyingErrors[0]
                let srError2 = underlyingError as? SRError
                let nsError2 = srError2 as? NSError
                
                if nsError2?.code as? Int != 8201 {
                    print("Error Getting SensorKit Authorization!")
                    return
                }
            }
        }
    }
    startMonitoring() // Continued in the next section.
}
```

The user will be prompted with a special Apple Research Permissions screen that will inform the user of any sensors being used. These sensors **MUST** be set via the `.entitlements` file within the XCode project.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>com.apple.developer.sensorkit.reader.allow</key>
	<array>
		<string>ambient-light-sensor</string>
	</array>
</dict>
</plist>
```

Once the user has given the application permissions, the user will never be prompted with this request again unless they revoke research permissions from the Settings app. 
Any subsequent updates to the application that results in permission changes will always prompt the user to give explicit consent to the collection of their data via SensorKit.

[!NOTE]
> Error code 8201 indicates that services are already authorized, which can be safely ignored.

### 3. Start Monitoring

Implementation for starting the sensor monitoring:

```swift
private func startMonitoring() {
    sensorReader.fetchDevices()
    sensorReader.startRecording()
    fetchSamples(device: device)
}

private func fetchSamples(device: SRDevice?) {
    guard let device = device else {
        print("No device found for this sensor")
        return
    }
    fetchRequest.device = device
    fetchRequest.from = SRAbsoluteTime.fromCFAbsoluteTime(_cf: 0)
    fetchRequest.to = SRAbsoluteTime.current()
    sensorReader.fetch(fetchRequest)
}
```

Here we make use of `SensorReader.startRecording()` to begin collecting sensor data while the application is in use. This data is saved and held on device for 24 hours before being released for use. This is done to give participants a short window of time to change their minds and revoke consent to the use of their data.

We also make use of `SensorReader.fetchDevices()` method to determine if the current device even has the necessary sensors we are looking for.
In our case we are checking for `.ambientLightSensor` as this is what was provided to the `SensorReader` when we initialized it.

Once we have our device list, we can begin constructing an `SRFetchRequest`. An `SRFetchRequest` sends a request to the system API responsible for managing secure sensor data storage, and returns all available samples that fit within the `SRFetchRequest.from` and `SRFetchRequest.to` range. Once the fetch request is constructed we can fetch our data by calling the `SensorReader.fetch(SRFetchRequest)` function. Our results will be sent to the 

### 4. Implement Delegate Methods

SensorKit relies on delegate methods for handling interaction with the system API responsible for managing the device sensors.
This means that our `Manager` class has access to the specific methods that will be called by the class deemed to be the delegate (also `Manager`).

Key delegate methods for handling sensor data:

- Simple delegate method implementation for getting the `[SRDevice]` from `SRSensorReader.fetchDevices()`.
```swift
func sensorReader(_ reader: SRSensorReader, didFetch devices: [SRDevice]) {
    // Here device is just a private class variable
    device = devices[0]
}
```

- Delegate method implementation for collecting all results from a `SRSensorReader.fetch(SRFetchRequest)` call and storing them in a private class array.
    - In this particular example you can notice that I filter out any results with <= 5 Lux. This is because SensorKit seriously tanks performance on iOS devices when loading stored samples! Knowing how to filter your results will speed up the loading process if you can get away with doing so.
    - I highly recommend storing your results in some kind of `private var` and using a getter method to access the results. Having this `published` is an awful idea as it locks the write access to the main thread which is terrible for reading thousands of samples from storage. 
    - Going from `published` to `private` resulted in a `~1600x performance increase (0.1fps -> ~320fps)`. This suggests an issue with Swift class `published` vars.

```swift
func sensorReader(_ reader: SRSensorReader, fetching fetchRequest: SRFetchRequest, didFetchResult result: SRFetchResult<AnyObject>) -> Bool {
    switch result.sample {
    case let lightSample as SRAmbientLightSample:
        if lightSample.lux.value > 5 {
            let date = result.timestamp.toCFAbsoluteTime()
            let res = SVILightSample(lux: lightSample.lux, 
                                   chromacity: lightSample.chromaticity, 
                                   placement: lightSample.placement, 
                                   date: date)
            samples.append(res)
        }
    default:
        print("Unhandled sample type: \(result.sample)")
        return false
    }
    return true
}
```

## Important Considerations

1. **Data Holding Period**: SensorKit implements a 24-hour holding period for sensor data. This means that data collected will only become available after 24 hours.

2. **Authorization Status**: Always check the authorization status before attempting to access sensor data:
```swift
func sensorReader(
    _ reader: SRSensorReader,
    didChange authorizationStatus: SRAuthorizationStatus
) {
    print("Sensor Reader Authorization State: \(authorizationStatus)")
}
```

3. **Error Handling**: Implement proper error handling for device fetching:
```swift
func sensorReader(_ reader: SRSensorReader, fetchDevicesDidFailWithError error: Error) {
    print("Error fetching devices: \(error)")
}
```

## Common Issues and Solutions

1. **Authorization Errors**: 

- The most common error (8201) indicates services are already authorized. This can be safely ignored.

2. **Data Availability**: 

- Remember that sensor data has a 24-hour holding period before it becomes available.

3. **Device Detection**: 

- Always check for valid device references before attempting to fetch samples.

4. **Poor Performance**: 

- As previously mentioned, using a getter method that returns the array of samples is ~1600x faster than accessing the array from the published class variable. I don't actually like using encapsulation, though evidently this works better and Swift is already a very OO language. If you find a better solution that allows for asynchronous reads (other than this one), let me know! The simple function below was enough for a huge performance increase, but I believe you could squeeze much more performance with something that had a bit more care put into it.

```swift
func getSamples() -> [SVILightSample] {
    if !samples.isEmpty {
        return samples
    } else {
        return []
    }
}
```

## Conclusion

Hopefully this article is more than enough info for anyone looking to get started with SensorKit for applied research projects. Now that you know how to setup SensorKit within a `Manager` class, I recommend checking out the documentation as you should have more than enough knowledge to quickly pick up on everything else. Best of luck!

See: [Apple Developer Documentation](https://developer.apple.com/documentation/sensorkit)

### Final Code Example

```swift
struct SVILightSample {
    var lux: Measurement<UnitIlluminance>
    var chromacity: SRAmbientLightSample.Chromaticity
    var placement: SRAmbientLightSample.SensorPlacement
    var date: CFAbsoluteTime
}

// This class extends SRSensorReaderDelegate making it the delegate object (NSObject) that is repeatedly referenced for any of OUR SensorKit queries.
// In this case we only care about Lux at the moment.
class LightManager: NSObject, ObservableObject, SRSensorReaderDelegate {
    // Having this published is an awful idea as it locks the write access to the main thread. We can circumvent this
    // by literally just making it private and writing a getter function. This results in like a 1600x performance increase :D
    private var samples: [SVILightSample] = []
    private var sensorReader = SRSensorReader(sensor: .ambientLightSensor)
    private var fetchRequest = SRFetchRequest()
    private var device = SRDevice()
    
    override init() {
        super.init()
        setupSensorKit()
    }
    
    // Assign delegate class to our sensorReader and then request auth
    private func setupSensorKit() {
        print("Setting up SensorKit...")
        sensorReader.delegate = self
        requestAuthorization()
    }
    
    // Retrieve our list of AmbientLightSamples
    func getSamples() -> [SVILightSample] {
        if !samples.isEmpty {
            return samples
        } else {
            return []
        }
    }
    
    // This function really only matters on first install and launch of the application.
    // Subsequent calls to this function will throw an Error
    // "Invalid authorization request. Requested services are already authorized: Error Domain=SRErrorDomain Code=8201 "(null)"
    // We can just ignore this kind of error as it means that we are already authorized.
    private func requestAuthorization() {
        if (sensorReader.authorizationStatus != SRAuthorizationStatus.authorized) {
            SRSensorReader.requestAuthorization(sensors: [.ambientLightSensor]) {  error in
                
                // We need to decompose this disgusting nested error type that Apple insists on working with
                // Error -> SRError -> NSError -> Error -> SRError -> NSError
                let srError = error as? SRError
                let nsError = srError as? NSError
                
                if (nsError?.code as? Int == 4) {
                    let underlyingError = nsError?.underlyingErrors[0]
                    let srError2 = underlyingError as? SRError
                    let nsError2 = srError2 as? NSError
                    
                    if nsError2?.code as? Int != 8201 {
                        print("Error Getting SensorKit Authorization!")
                        return
                    }
                }
            }
        }
        startMonitoring()
    }
    
    // This function call will begin recording for our dataset.
    // It will also request all available samples that have been released after the 24 hour holding period.
    private func startMonitoring() {
        print("Starting monitoring...")
        print("\(sensorReader.authorizationStatus)")
        sensorReader.fetchDevices()
        sensorReader.startRecording()
        print("Started recording Lux Sensor Readings")
        
        fetchSamples(device: device)
    }
    
    // Fetch all samples from the beginning of time from our SRDevice
    private func fetchSamples(device: SRDevice?) {
            guard let device = device else {
                print("No device found for this sensor")
                return
            }
            fetchRequest.device = device
            fetchRequest.from = SRAbsoluteTime.fromCFAbsoluteTime(_cf: 0)
            fetchRequest.to = SRAbsoluteTime.current()
            sensorReader.fetch(fetchRequest)
        }
    
    // Delegate method (mostly for debugging)
    func sensorReader(_ reader: SRSensorReader, didFetch devices: [SRDevice]) {
        device = devices[0]
        print("Fetch Called!")
    }
    
    // Delegate method (mostly for debugging)
    func sensorReader(_ reader: SRSensorReader, fetchDevicesDidFailWithError error: Error) {
            print("Error fetching devices: \(error)")
        }
        
    // Called when we have a valid SRDevice ID and our SRFetchRequest is considered successful!
    func sensorReader(_ reader: SRSensorReader, fetching fetchRequest: SRFetchRequest, didFetchResult result: SRFetchResult<AnyObject>) -> Bool {
        switch result.sample {
        case let lightSample as SRAmbientLightSample:
            if lightSample.lux.value > 5 {
                let date = result.timestamp.toCFAbsoluteTime()
                let res = SVILightSample(lux: lightSample.lux, chromacity: lightSample.chromaticity, placement: lightSample.placement, date: date)
                samples.append(res)
            }
        default:
            print("Unhandled sample type: \(result.sample)")
            return false
        }
        return true
    }
    
    // Delegate method
    // Realistically this function will only be called on the first install of the application.
    func sensorReader(
        _ reader: SRSensorReader,
        didChange authorizationStatus: SRAuthorizationStatus
    )
    {
        print("Sensor Reader Authorization State: \(authorizationStatus)")
    }
}


```
