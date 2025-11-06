// Stepper motor on Wokwi!

#include <Stepper.h>
#include <math.h>

const int stepsPerRevolution = 360;  // change this to fit the number of steps per revolution
// for your motor
float x = 120;
float r = 3;
float K = x/(2*M_PI*r);

// initialize the stepper library on pins 8 through 11:
Stepper myStepper(stepsPerRevolution, 8, 9, 10, 11);

void setup() {
  // set the speed at 60 rpm:
  myStepper.setSpeed(60);
  // initialize the serial port:
  Serial.begin(9600);
}

void loop() {
  // step one revolution  in one direction:
  Serial.println("clockwise");
  myStepper.step(K*stepsPerRevolution);
  delay(500);
  /*
  // step one revolution in the other direction:
  Serial.println("counterclockwise");
  myStepper.step(-stepsPerRevolution);
  delay(500);
  */
}