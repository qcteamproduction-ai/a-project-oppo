#include <Arduino.h>

// Pin definitions - SESUAIKAN DENGAN SETUP ANDA
const int IR_SENSOR_PIN = 2;   // Digital pin untuk IR sensor
const int RELAY_PIN = 4;       // Digital pin untuk relay control
const int STATUS_LED_PIN = 13; // Built-in LED untuk status

// Variables
bool lastIRState = false; 
bool currentRelayState = true;  // true = relay ON (normal), false = relay OFF (defect)
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 100;  // 100ms debounce

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Initialize pins
  pinMode(IR_SENSOR_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  
  // Initial states
  digitalWrite(RELAY_PIN, HIGH);     // Relay ON initially (normal state)
  digitalWrite(STATUS_LED_PIN, LOW); // LED OFF initially
  currentRelayState = true;
  
  Serial.println("ESP32 Quality Control System Ready");
  Serial.println("Commands: STATUS, RELAY_ON, RELAY_OFF, RESET");
  Serial.println("Pin Configuration:");
  Serial.println("IR Sensor: Pin 2");
  Serial.println("Relay: Pin 4");
  Serial.println("Status LED: Pin 13");
}

void loop() {
  // Read IR sensor with debouncing
  bool currentIRState = digitalRead(IR_SENSOR_PIN);
  
  if (currentIRState != lastIRState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    // State has been stable for debounce period
    if (currentIRState != lastIRState) {
      if (currentIRState) {
        // Object detected
        digitalWrite(STATUS_LED_PIN, HIGH);
        Serial.println("IR:DETECTED");
      } else {
        // No object detected
        digitalWrite(STATUS_LED_PIN, LOW);
        Serial.println("IR:NOT_DETECTED");
      }
      lastIRState = currentIRState;
    }
  }
  
  // Handle serial commands from PC
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "STATUS") {
      // Send current status
      if (digitalRead(IR_SENSOR_PIN)) {
        Serial.println("IR:DETECTED");
      } else {
        Serial.println("IR:NOT_DETECTED");
      }
      
      if (currentRelayState) {
        Serial.println("RELAY:ON");
      } else {
        Serial.println("RELAY:OFF");
      }
    }
    else if (command == "RELAY_ON") {
      // Turn relay ON (normal operation)
      digitalWrite(RELAY_PIN, HIGH);
      currentRelayState = true;
      Serial.println("RELAY:ON_OK");
    }
    else if (command == "RELAY_OFF") {
      // Turn relay OFF (defect detected)
      digitalWrite(RELAY_PIN, LOW);
      currentRelayState = false;
      Serial.println("RELAY:OFF_OK");
    }
    else if (command == "RESET") {
      // Reset to normal state
      digitalWrite(RELAY_PIN, HIGH);
      digitalWrite(STATUS_LED_PIN, LOW);
      currentRelayState = true;
      Serial.println("ESP32:RESET_OK");
    }
    else if (command == "PING") {
      // Ping response for connection check
      Serial.println("ESP32:PONG");
    }
    else {
      Serial.println("ESP32:UNKNOWN_COMMAND");
    }
  }
  
  delay(50);  // Small delay to prevent flooding
}