const int PULSE_PIN = 2;    
const int DIR_PIN = 4;   
const int IR_SENSOR_PIN = 5; 

int pulseDelay = 500;      
const int SLOW_SPEED = 7000; 
const long DETECTION_DURATION = 10000; 

unsigned long lastDetectionTime = 0; 
bool isSlowMode = false;    

void setup() {
  pinMode(PULSE_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(IR_SENSOR_PIN, INPUT); 
  digitalWrite(DIR_PIN, HIGH); 
}

void loop() {
  if (digitalRead(IR_SENSOR_PIN) == LOW) {
    lastDetectionTime = millis(); 
    isSlowMode = true; 
  }

  if (isSlowMode && (millis() - lastDetectionTime >= DETECTION_DURATION)) {
    isSlowMode = false; 
  }

  int currentDelay = isSlowMode ? SLOW_SPEED : pulseDelay;

  digitalWrite(PULSE_PIN, HIGH);
  delayMicroseconds(currentDelay / 2);
  digitalWrite(PULSE_PIN, LOW);
  delayMicroseconds(currentDelay / 2);
}