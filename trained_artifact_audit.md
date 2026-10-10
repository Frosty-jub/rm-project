# ตรวจไฟล์โมเดลที่ฝึกจริง (10 ตุลาคม 2026)

ขอบเขต: รายงานนี้ตรวจ checkpoint ที่ฝึกหลังการกู้โฟลเดอร์ เป็นการตรวจอิสระจากการคำนวณใน runner โดยโหลดไฟล์ `model.pt` กลับมาและทำนายจากไฟล์ Dataset A ดิบผ่าน `scripts/audit_saved_suite.py` ไม่ใช่การรีวิว checkpoint โดย GPT-6 Astra; รายงาน Astra เดิมที่ `astra_review.md` ตรวจโค้ดก่อนการฝึกจริง

สำหรับทุกคู่ ตัวตรวจยืนยัน SHA-256 ของ checkpoint และไฟล์ข้อมูลตาม `result.json`, ใช้สถิติ preprocessing และน้ำหนัก ensemble ที่บันทึกไว้, ตรวจคะแนน source/target และจำนวนคำทำนายทุกคลาสกับผลที่ runner บันทึก, ตรวจผลรวม score ตามผลรวมน้ำหนัก (สูตร raw weights ไม่จำเป็นต้องรวมเป็น 1), ตรวจค่าใน CSV และ epoch สุดท้ายของ history การรันซ้ำนี้ทำบน CPU จึงยอมความต่าง float32 ของ accuracy ไม่เกินเศษหนึ่งส่วนสิบของตัวอย่างหนึ่งตัว

| Artifact | จำนวนคู่ที่ตรวจ | ผล |
|---|---:|---|
| `runs/revised/pilot_domain` | 3/3 | ผ่าน |
| `runs/revised/pilot_signedlog_domain` | 3/3 | ผ่าน |
| `runs/revised/pilot_source_row_l2` | 3/3 | ผ่าน |
| `runs/revised/full_signedlog_1_to_2` | 1/1 | ผ่าน |
| `runs/revised/equation_sensitivity` | 1/1 | ผ่าน; raw weights ให้ score sum > 1 ตามสูตร |
| `runs/revised/center_sum_1_to_2` | 1/1 | ผ่าน; reuse pretraining checksum ตรง |
| `runs/revised/mixed_mmd_1_to_2` | 1/1 | ผ่าน; reuse pretraining checksum ตรง |
| `runs/revised/pilot_signedlog_full_suite` | 17/17 | ผ่าน; 9 long และ 9 short protocol rows |
| `runs/revised/pilot_row_l2_full_suite_resume` | 17/17 | ผ่าน; 9 long และ 9 short protocol rows |
| `runs/revised/full_signedlog_domain_resume3` | 17/17 | ผ่าน; 9 long และ 9 short protocol rows |

โค้ดแกนที่สร้างโดย GPT-6 Sol และตรวจโดย GPT-6 Astra ไม่ถูกแก้หลังการกู้: `mpc_cnn/revised.py` SHA-256 `ede6262ca2833653a270561b407741abaa8d65ba2ea46a916ac2f1ead40ac8fb`, `scripts/run_revised_suite.py` SHA-256 `d2cb4fe3c6362f7875416011f91972ec0f742f2a97eaecc6ae15b1a58d088b95`, `tests/test_revised.py` SHA-256 `f3585552be88f9fd5c41cf786e2a61c4523fe51a933bfe70d438955f32e5a7a8` รายงาน Astra เป็นการตรวจ implementation ว่ารันทดลองได้ ไม่ใช่การยืนยันว่าจะทำซ้ำ accuracy ของ paper
