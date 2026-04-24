# ESP32-CAM Currency Recognition Circuit Diagram

![Generated Circuit Diagram](/Users/doni/.gemini/antigravity/scratch/Currency_Recognition_Project/circuit_diagram.png)

This document contains the pin mapping and circuit diagram for the project based on the BOM (DFPlayer Mini + PAM8403).

## 1. Pin Mapping Table
| Component | ESP32-CAM Pin | Type | Notes |
| :--- | :--- | :--- | :--- |
| **IR Sensor** | GPIO 13 | Input | Digital Trigger (Active Low/High) |
| **Push Button** | GPIO 12 | Input | Digital Trigger (Active High w/ Pulldown) |
| **DFPlayer TX** | GPIO 15 | Output | Connect to DFPlayer RX via 1kΩ Resistor |
| **DFPlayer RX** | GPIO 14 | Input | Connect to DFPlayer TX |
| **White LED** | GPIO 4 | Output | External LED (Shared with Flash pin) |
| **FTDI TX** | U0R (GPIO 3) | Input | Connect to ESP32-CAM RX for programming |
| **FTDI RX** | U0T (GPIO 1) | Output | Connect to ESP32-CAM TX for programming |
| **Flash Mode Drop** | GPIO 0 | Input | Connect to GND when programming via FTDI |
| **Power (5V)** | 5V Pin | Power | From MT3608 Boost Converter |
| **Ground** | GND | Power | Common Ground |
| **Filter Capacitor** | 5V & GND | Power | 470µF - 1000µF Electrolytic Cap across rails to prevent brownouts |
## 2. Mermaid Circuit Diagram
```mermaid
graph LR

    %% Power Delivery
    BAT[18650 Battery] -->|B+ to B+| TP[TP4056 Charger B_IN]
    BAT -->|B- to B-| TP
    
    TP -->|OUT+ to VIN+| MT[MT3608 Boost CONV]
    TP -->|OUT- to VIN-| MT
    
    MT -->|VOUT+  --->| VCC((Common 5V Line))
    MT -->|VOUT-  --->| GND((Common Ground Line))
    
    %% Power Filtration
    CAP[470µF - 1000µF Capacitor]
    VCC -->|+ Anode (Long Leg)| CAP
    GND -->|- Cathode (Short Leg, Stripe)| CAP
    
    %% Core Controller
    ESP[ESP32-CAM]
    VCC -->|to 5V Pin| ESP
    GND -->|to GND Pin| ESP
    
    %% Inputs and Sensors
    IR[IR Sensor]
    VCC -->|to VCC| IR
    GND -->|to GND| IR
    IR -->|OUT pin  --->  GPIO 13 pin| ESP
    
    PB[Push Button]
    VCC -->|to Leg 1| PB
    PB -->|from Leg 2 ---> GPIO 12 pin| ESP
    
    %% Programming (FTDI)
    FTDI[FTDI USB-to-TTL]
    FTDI -->|GND pin ---> GND pin| ESP
    FTDI -->|TX pin  ---> GPIO 3 / U0R pin| ESP
    FTDI -->|RX pin  ---> GPIO 1 / U0T pin| ESP
    GND -.->|Temporary Jumper ---> GPIO 0 pin| ESP
    
    %% Audio System
    DF[DFPlayer Mini]
    VCC -->|to VCC| DF
    GND -->|to GND| DF
    ESP -->|GPIO 15 pin ---> [1kΩ Resistor] ---> RX pin| DF
    DF -->|TX pin  ---> GPIO 14 pin| ESP
    
    PAM[PAM8403 Amplifier]
    VCC -->|to 5V| PAM
    GND -->|to GND| PAM
    DF -->|DAC_R pin ---> R Input pin| PAM
    DF -->|DAC_L pin ---> L Input pin| PAM
    
    SPK[3W Speaker]
    PAM -->|L/R Output +/- pins ---> terminal pins| SPK
    
    %% Optics
    LED[White Ext LED]
    ESP -->|GPIO 4 pin ---> + Anode| LED
    LED -->|- Cathode --->| GND

```
