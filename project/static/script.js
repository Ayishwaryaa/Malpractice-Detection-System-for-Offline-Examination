// Set objects to keep track of previous states
let previousBLEDevices = new Set();
let previousWiFiNetworks = new Set();

// Displays a toast notification
function showNotification(message) {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;

    container.appendChild(notification);

    // Fade out and remove after 4s
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            if (notification.parentNode) {
                container.removeChild(notification);
            }
        }, 500);
    }, 4000);
}

function startDetection() {
    const width = parseFloat(document.getElementById('roomWidth')) || 0;
    const length = parseFloat(document.getElementById('roomLength')) || 0;

    let query = '';
    if (width > 0 && length > 0) {
        query = `?width=${width}&length=${length}`;
    } else {
        query = `?width=${0}&length=${0}`; // sending 0,0 to backend
    }

    fetch(`/start${query}`)
        .then(response => response.json())
        .then(data => {
            console.log(data.status);
            showNotification("Detection Started!");
            start();
        })
        .catch(error => {
            console.error("Error starting detection:", error);
            showNotification("Error Starting Detection ❌");
        });
}

function getResult() {
    fetch('/get_result')
        .then(response => response.json())
        .then(data => {
            document.getElementById('output').textContent = `${data}`;
            console.log(`${data}`);
        })
        .catch(error => {
            console.error("Error fetching result:", error);
            document.getElementById("output").textContent = "Error fetching result.";
        });
}



function stopDetection() {
    fetch('/stop')
        .then(response => response.json())
        .then(data => {
            
            console.log(data.status);
            showNotification("Detection Stopped!");
            window.location.href = "/results";
            stop();
        })
        .catch(error => console.error("Error starting detection:", error));


}


function fetchBLEDevices() {
    fetch('/ble_devices')
        .then(response => response.json())
        .then(data => {
            const bleList = document.getElementById('bleList');
            bleList.innerHTML = '';

            const currentDevices = new Set();

            data.devices.forEach(device => {
                const identifier = `${device[0]} (${device[1]})`; // (device name , device address )Converting to string
                currentDevices.add(identifier);

                const li = document.createElement('li');
                li.textContent = identifier;
                bleList.appendChild(li);

                if (!previousBLEDevices.has(identifier)) {
                    showNotification(`New Bluetooth Device Detected: ${identifier}`);
                }
            });

            // Merge sets instead of overwriting
            for (let id of currentDevices) {
                previousBLEDevices.add(id);
            }
        })
        .catch(error => console.error("BLE fetch error:", error));
}


function fetchWiFiNetworks() {
    fetch('/wifi_networks')
        .then(response => response.json())
        .then(data => {
            const wifiList = document.getElementById('wifiList');
            wifiList.innerHTML = '';

            const currentNetworks = new Set();

            data.networks.forEach(network => {
                const identifier = `${network.ssid} (${network.signal})`; // Converting to string
                currentNetworks.add(identifier);

                const li = document.createElement('li');
                li.textContent = identifier;
                wifiList.appendChild(li);

                if (!previousWiFiNetworks.has(identifier)) {
                    showNotification(`New WiFi Network Detected: ${network.ssid}`);
                }
            });

            // Merge sets instead of overwriting
            for (let id of currentNetworks) {
                previousWiFiNetworks.add(id);
            }
        })
        .catch(error => console.error("WiFi fetch error:", error));
}


setInterval(fetchBLEDevices, 3000);
setInterval(fetchWiFiNetworks, 3000);
setInterval(getResult, 3000);