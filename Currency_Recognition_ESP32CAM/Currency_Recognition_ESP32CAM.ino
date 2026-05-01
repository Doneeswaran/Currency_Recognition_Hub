/*
 * ============================================================
 *  Currency Recognition Hub - Unified ESP32-CAM Firmware
 *  - Core 1 (Main Loop): Camera capture → POST to Flask server
 *  - Core 0 (BG Task):   Poll audio bridge → Play via I2S
 * ============================================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include "driver/i2s.h"

// ----- WiFi Credentials -----
const char* ssid     = "Doni";
const char* password = "Thambu2005";

// ----- Server URLs -----
// NOTE: Update to your computer's actual local IP address
const char* recognizeUrl = "http://192.168.1.5:5001/recognize";
const char* audioUrl     = "http://192.168.1.5:8000/poll_audio";

// ----- Flash LED -----
#define FLASH_LED_PIN 4

// ----- Camera Pins (AI Thinker ESP32-CAM) -----
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

// ----- I2S Speaker Pins (SD Card pins, free when no SD is used) -----
#define I2S_DOUT 13
#define I2S_BCLK 14
#define I2S_LRC  15

// ----- FreeRTOS Task Handle -----
TaskHandle_t audioTaskHandle = NULL;

// ===================================================================
//  CORE 0 TASK: Audio Polling
//  Continuously polls the audio bridge server.
//  If the server returns audio (HTTP 200), play it via I2S.
//  If nothing new (HTTP 204), wait and try again.
// ===================================================================
void audioTask(void* parameter) {
  Serial.println("[Audio Core 0] Audio polling task started.");

  while (true) {
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(audioUrl);
      http.setTimeout(3000);

      int httpCode = http.GET();

      if (httpCode == 200) {
        // New currency detected — stream the audio file
        Serial.println("[Audio] Currency detected! Playing voice announcement...");
        WiFiClient* stream = http.getStreamPtr();
        uint8_t buffer[512];

        // *** FIX: Skip the 44-byte WAV file header ***
        // WAV files begin with RIFF/WAVE metadata before actual PCM audio.
        // Sending these bytes to I2S causes a loud click/noise pop at the start.
        uint8_t wav_header[44];
        stream->readBytes(wav_header, 44);

        while (http.connected()) {
          int len = stream->readBytes(buffer, sizeof(buffer));
          if (len > 0) {
            size_t written;
            i2s_write(I2S_NUM_0, buffer, len, &written, portMAX_DELAY);
          } else {
            break;
          }
        }
        Serial.println("[Audio] Playback complete.");
      } else if (httpCode == 204) {
        // No new currency — silently wait
      } else {
        Serial.printf("[Audio] Poll error: HTTP %d\n", httpCode);
      }

      http.end();
    }

    // Poll every 500ms
    vTaskDelay(500 / portTICK_PERIOD_MS);
  }
}

// ===================================================================
//  SETUP: WiFi, Camera, I2S, FreeRTOS Task
// ===================================================================
void setup() {
  Serial.begin(115200);

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // --- Connect to WiFi ---
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected! IP: " + WiFi.localIP().toString());

  // --- Initialize I2S for Audio Output ---
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = 22050,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 256
  };

  i2s_pin_config_t i2s_pins = {
    .bck_io_num   = I2S_BCLK,
    .ws_io_num    = I2S_LRC,
    .data_out_num = I2S_DOUT,
    .data_in_num  = I2S_PIN_NO_CHANGE
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &i2s_pins);
  Serial.println("I2S Speaker Ready.");

  // --- Initialize Camera ---
  camera_config_t cam_config;
  cam_config.ledc_channel = LEDC_CHANNEL_0;
  cam_config.ledc_timer   = LEDC_TIMER_0;
  cam_config.pin_d0       = Y2_GPIO_NUM;
  cam_config.pin_d1       = Y3_GPIO_NUM;
  cam_config.pin_d2       = Y4_GPIO_NUM;
  cam_config.pin_d3       = Y5_GPIO_NUM;
  cam_config.pin_d4       = Y6_GPIO_NUM;
  cam_config.pin_d5       = Y7_GPIO_NUM;
  cam_config.pin_d6       = Y8_GPIO_NUM;
  cam_config.pin_d7       = Y9_GPIO_NUM;
  cam_config.pin_xclk     = XCLK_GPIO_NUM;
  cam_config.pin_pclk     = PCLK_GPIO_NUM;
  cam_config.pin_vsync    = VSYNC_GPIO_NUM;
  cam_config.pin_href     = HREF_GPIO_NUM;
  cam_config.pin_sscb_sda = SIOD_GPIO_NUM;
  cam_config.pin_sscb_scl = SIOC_GPIO_NUM;
  cam_config.pin_pwdn     = PWDN_GPIO_NUM;
  cam_config.pin_reset     = RESET_GPIO_NUM;
  cam_config.xclk_freq_hz = 8000000;
  cam_config.pixel_format = PIXFORMAT_JPEG;
  cam_config.frame_size   = FRAMESIZE_VGA;
  cam_config.jpeg_quality = 15;
  cam_config.fb_count     = 2;

  esp_err_t err = esp_camera_init(&cam_config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    delay(1000);
    ESP.restart();
  }

  // Tune sensor for better image quality
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
  }
  Serial.println("Camera Ready!");

  // --- Start Audio Task on Core 0 ---
  // The camera loop runs on Core 1 (Arduino default).
  // We pin the audio task to Core 0 so they never conflict.
  xTaskCreatePinnedToCore(
    audioTask,        // Task function
    "AudioTask",      // Task name
    8192,             // Stack size (bytes)
    NULL,             // Parameters
    1,                // Priority
    &audioTaskHandle, // Task handle
    0                 // Run on Core 0
  );

  Serial.println("\n=== Currency Recognition Hub Started ===");
  Serial.println("Core 1: Camera scanning → Flask server");
  Serial.println("Core 0: Audio polling  → Bridge server");
  Serial.println("========================================");
}

// ===================================================================
//  LOOP (CORE 1): Camera Capture and Recognition
//  Grabs a frame, sends to Flask server, logs result to Serial.
//  Audio is handled independently on Core 0 — no blocking here.
// ===================================================================
void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[Camera] Capture failed. Retrying...");
    delay(300);
    return;
  }

  HTTPClient http;
  http.begin(recognizeUrl);
  http.addHeader("Content-Type", "image/jpeg");
  http.setConnectTimeout(8000);
  http.setTimeout(15000);

  int code = http.POST(fb->buf, fb->len);

  if (code > 0) {
    String response = http.getString();
    if (response != "unknown" && response != "error") {
      Serial.println(">>> NOTE DETECTED: Rs." + response);
    }
  } else {
    Serial.printf("[Camera] Network Error: %s\n", http.errorToString(code).c_str());
  }

  http.end();
  esp_camera_fb_return(fb);

  delay(100); // ~10 FPS capture rate
}
