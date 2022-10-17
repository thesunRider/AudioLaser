#include <TimerOne.h>
#define speaker_out 10
#define enable_speaker 11
#define pulse_ip 7
#define ampl_detect A0

#define sample_delay 50
#define sample_no 20
#define freq_value_min  50
#define freq_value_max  2000
#define duty_cycle_min  20
#define duty_cycle_max  80

#define frequency_skip 6
#define duty_cycle_skip 5
 

float time_period;

void setup(){
  Serial.begin(9600);
  
  Timer1.initialize(150000);
  Timer1.pwm(speaker_out, (float)(50.00 / 100) * 1023);
  
  pinMode(pulse_ip,INPUT);
  pinMode(ampl_detect,INPUT);
  pinMode(enable_speaker,OUTPUT);
  digitalWrite(enable_speaker,LOW);

}

void loop(){
  test_all();
}

void test_all(){
 float result[6] = {0};

 Serial.println("FREQ TMPRD DUTY MONTIME MOFFTIME MDUTY MFREQ MTMPRD MAMP");
 for (int i = freq_value_min;i <= freq_value_max;i+=frequency_skip){
  time_period = 1000.00 / ((float)i);

  setoutput(time_period,50); //50 % duty cycle
  delay(2000);
  
 }

}

void setoutput(float time_period,float duty_cycle ){
  Timer1.setPeriod((float) 1000 * time_period);
  Timer1.setPwmDuty(speaker_out, (float)((duty_cycle) / 100) * 1023);
  Timer1.restart();
}
