int Sensor1Val = 0; // variable to hold the value from the red sensor
int Sensor2Val = 0; // variable to hold the value from the green sensor
int Sensor3Val = 0; // variable to hold the value from the blue sensor

void setup() {
  // initialize serial communications with baud rate 9600
  Serial.begin(9600);
}

void loop() {
  // Read the sensors first:

  // read the value from the signal pin of the sensor board
  Sensor1Val = analogRead(A0);
  // intermediate delays to allow the Analog-Digital Convertor to settle
  delay(5);
  Sensor2Val = analogRead(A1);
  delay(5);
  Sensor3Val = analogRead(A2);

  // forward values to the Serial Monitor with delimiter encoding
  // could use binary encoding rather than plain text, but may cause buffer issues on the other side
  Serial.print(Sensor1Val);
  Serial.print('-');
  Serial.print(Sensor2Val);
  Serial.print('-');
  Serial.print(Sensor3Val);
  Serial.print('!');
  delay(5);
}
