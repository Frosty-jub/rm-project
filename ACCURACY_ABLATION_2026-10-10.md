# ตรวจสาเหตุ accuracy ลดของ MPC-CNN (10 ตุลาคม 2026)

## คำตอบที่ระบุได้จากการรัน

**จุดหลักในงาน Batch 4→5 คือแรงของ multi-level center loss ระหว่าง formal training** ที่ `mpc_cnn/revised.py:443` (`loss = ce + center_coefficient * cl`) โดย `cl` รวมระยะจาก feature ทั้ง 3 ระดับไปยัง class center ที่ `mpc_cnn/model.py:78` และ center ถูกอัปเดตจากข้อมูล source ที่ `mpc_cnn/revised.py:453` เท่านั้น การลดแรงของพจน์นี้เพียงตัวเดียวเพิ่ม target accuracy แบบต่อเนื่องในคู่ 4→5:

| ค่า `center_multiplier` | ค่าสัมประสิทธิ์จริงหลัง warmup | Source accuracy | Target accuracy | เทียบค่าเดิม |
|---:|---:|---:|---:|---:|
| 0.01 (ค่าเดิม) | 0.020 | 100.00% | 56.85% | — |
| 0.001 (1/10) | 0.002 | 100.00% | 67.51% | +10.66 จุด |
| 0 (ปิด) | 0 | 100.00% | 80.20% | +23.35 จุด |

หลังตรวจ [Supporting Information](https://acs.figshare.com/articles/journal_contribution/Gas_Sensor_Drift_Compensation_Using_Semi-Supervised_Ensemble_Classifiers_with_Multi-Level_Features_and_Center_Loss/28752293) ตาราง S3 เพิ่มเติม พบว่างาน 4→5 ระบุ 800 epochs, learning rate 8×10⁻⁴, batch size 160 และ `w=2` ซึ่งสามค่าแรกตรงกับรันนี้ แต่สูตรค่าเดิมของเราใช้ `center_multiplier=0.01` ทำให้ค่าสัมประสิทธิ์ center จริงเท่ากับ 0.02 ไม่ใช่ 2. จึงทดลอง `center_multiplier=1` เพื่อให้สัมประสิทธิ์เป็น **2** โดยคง loss แบบ `mean` และ checkpoint เดิมไว้ ผลคือ source **91.93%**, target **34.01%** ที่ epoch 800 และ replay checkpoint ผ่าน การยกค่า `w` จาก SI มาใส่ใน implementation ปัจจุบันโดยตรงจึงทำให้แย่ลงอีก ผลนี้ **ไม่ใช่** การรันสูตร paper เต็ม เพราะ Eq. 8 ของ PDF หลักเขียน center loss แบบผลรวม ขณะที่รันใหม่นี้ยังใช้ mean และรายละเอียด preprocessing/การเริ่ม centers ยังเป็นของเรา การเปลี่ยน `sum` พร้อม `w=2` จะเพิ่มสเกล penalty มากกว่าเดิมอีก แต่ยังไม่ได้ทดสอบตรง ๆ

ทั้งสามรันใช้ UCI batch 4/5, seed 42, signed-log และ domain z-score เดียวกัน, สถาปัตยกรรมและ optimizer เดียวกัน, 600 pretraining epochs, 800 formal epochs, RTX 2060/PyTorch 2.7.1+cu126, และ **โหลด pretrained checkpoint ไฟล์เดียวกัน** (SHA-256 `f5e204f8e3b37f2d0b1805ddfad01f37f6217afee48fa4797e0a25cf42501518`) ต่างกันเฉพาะ `center_multiplier` ใน formal stage ผลทั้งหมดเป็น final epoch ไม่มีการเลือก checkpoint ด้วย label ของ target

## จุดที่ accuracy หายและกลไก

1. **ทุกหัวจำแนกเสียพร้อมกัน ไม่ใช่แค่ ensemble fusion.** สำหรับ Batch 4→5 รุ่นค่าเดิม accuracy ของหัวระดับ 1/2/3 คือ 58.38%/56.35%/56.35%; ensemble 56.85% และทั้งสามหัวมี recall ของ target class index 4 เป็น 0% ขณะที่รุ่นปิด center ให้หัว 77.66%/77.66%/79.70%; ensemble 80.20% จึงเป็นการเปลี่ยน representation และตัวจำแนกระหว่างการฝึก ไม่ใช่ความผิดพลาดจากการเฉลี่ยความน่าจะเป็นตอนทดสอบเพียงอย่างเดียว น้ำหนัก ensemble รุ่นเดิมคือ 0.327/0.336/0.337 ใกล้ค่าเฉลี่ยเท่ากัน
2. **คลาสเป้าหมายหนึ่งคลาสหายไป.** Batch 4 มี source class index 4 เพียง 12 ตัวอย่าง; Batch 5 มี target class เดียวกัน 63 ตัวอย่าง รุ่นค่าเดิมทำนายคลาสนี้ถูก 0/63 และทำนายเป็นคลาส index 0 จำนวน 56 ตัวอย่าง รุ่นลดแรง center เหลือ 1/10 ทายถูก 6/63; รุ่นปิด center ทายถูก 27/63 การฝึกบังคับ source feature ให้เกาะ class center แต่ไม่ได้ใช้ target labels เพื่อวาง center จึงมีเหตุผลที่ class center จาก source จะไม่รองรับ drift ของคลาสนี้ นี่เป็น **คำอธิบายเชิงกลไกที่สอดคล้องกับผล**; การทดลองนี้ยังแยกไม่ได้ว่าการเริ่ม center, สูตร update, หรือความแรงของ penalty ภายในพจน์นี้มีส่วนเท่าไร
3. **ฝึกต่อแล้ว target แย่ลง ทั้งที่ source สมบูรณ์.** รุ่น center ค่าเดิม Batch 4→5 target จากประมาณ 74% ที่ formal epoch 100 เหลือ 56.85% ที่ epoch 800 โดย source 100% ตลอดช่วงที่บันทึก; ที่ epoch 100 และ 800 ค่า center penalty หลังคูณสัมประสิทธิ์อยู่ราว 2.5 และ 2.2 เท่าของ CE ตามลำดับ รุ่นปิด center ที่ epoch 100/800 ได้ 83.25%/80.20% จึงชี้ว่าการ optimize ตาม source และ center loss ต่อเนื่องกำลังทำลายการโอนย้ายในคู่นี้ การใช้ epoch 100 เป็นโมเดลจริงโดยเลือกจาก target label จะผิด protocol; ตัวเลขระหว่างทางใช้วิเคราะห์เท่านั้น

## ไม่ใช่ทุกงานที่ควรปิด center loss

งาน Batch 1→2 ใช้ pretrained checkpoint เดิมของคู่นั้นและเปลี่ยนเฉพาะ formal center: ค่าเดิมได้ **73.55%**, ปิด center ได้ **70.90%** (ลด 2.65 จุด) ดังนั้น center loss ช่วยบางงานและทำร้ายบางงาน ผลด้านบนระบุสาเหตุเฉพาะคู่ 4→5 ภายใต้ recipe นี้ ไม่ใช่ข้อพิสูจน์ว่า center loss ใน paper ผิดหรือควรลบทิ้งทุกกรณี

อีกความเสี่ยงที่วัดแบบคุมตัวแปรแล้วคือ **สเกลของ center loss**: คู่ 1→2 แบบ `mean` ได้ 73.55% แต่เปลี่ยนเป็น `sum` โดยใช้ coefficient เดิมได้ 54.66% (**−18.89 จุด**) เนื่องจาก `sum` ขยาย penalty ประมาณตามขนาด minibatch (งานนี้ 90) ในขณะที่ CE ยังเป็นค่าเฉลี่ย การเปลี่ยน reduction ให้ตรง Eq. 8 ของ paper ต้องปรับ coefficient ให้สอดคล้อง ไม่ควรสลับโหมดแล้วใช้ตัวเลขเดิม

MMD weight แบบคำนวณ source/target คนละ train/eval mode ทำให้เกิดระยะปลอมแม้ input เหมือนกัน แต่ one-factor run คู่ 1→2 ใน recipe นี้ลด accuracy จาก 73.55% เป็น 73.47% เท่านั้น (**−0.08 จุด**); จึงไม่ใช่สาเหตุหลักที่พิสูจน์ได้ของความต่างระดับ 19–23 จุดในงานที่วัด ส่วน ensemble คู่ 4→5 แพ้หัวเดี่ยวที่ดีที่สุดเพียง 1.52 จุด และการเลือกหัวดีที่สุดโดยดู target label เป็นเพียง oracle diagnostic

## ขอบเขตและความต่างจาก paper

- การรันนี้เป็น modified MPC-CNN recipe ที่มีสมมติฐานของเรา เช่น signed-log, domain-wise z-score, CE+MMD pretraining, center warmup, normalized ensemble weights และ Gaussian bandwidth แบบ median-differentiable จึงไม่ใช่การยืนยันว่าตรง implementation ที่ผู้เขียนใช้ทุกจุด
- Paper รายงาน Batch 4→5 = 94.71% และ Batch 1→2 = 85.06%; แม้ปิด center ใน 4→5 ยังต่ำกว่า paper 14.51 จุด ข้อมูลนี้ยังไม่ระบุสาเหตุของช่องว่างที่เหลือแน่ชัด
- เป็นการเปรียบเทียบ seed เดียวและสองคู่หลัก จึงไม่ประมาณความแปรปรวนข้าม seed และไม่สรุปผลทั้ง 17 task จากสองคู่นี้
- Target labels ใช้คำนวณ accuracy/diagnostic หลังการฝึกและบันทึก learning curve เท่านั้น ไม่ใช้กับ gradient, class centers, early stopping หรือเลือก final checkpoint; การเลือก recipe หลายครั้งจากผล target ในงานก่อนทำให้ตัวเลขดีที่สุดเป็น exploratory ไม่ใช่ blind test

## ไฟล์ตรวจสอบและวิธีรันซ้ำ

- Baseline: `runs/revised/full_signedlog_domain_resume3/batch4_to_5/result.json`, `runs/revised/full_signedlog_domain_resume3/batch1_to_2/result.json`
- Ablation: `runs/revised/ablation_full_no_center_4_to_5/`, `runs/revised/ablation_full_low_center_4_to_5/`, `runs/revised/ablation_full_no_center_1_to_2/`
- ทดสอบค่าน้ำหนักตาม SI ภายใต้ loss แบบปัจจุบัน: `configs/ablation_full_si_weight_mean.json`, `runs/revised/ablation_full_si_weight_mean_4_to_5/` (audited)
- Confusion/replay: `runs/revised/ablation_full_no_center_4_to_5/confusion.json`, `runs/revised/ablation_full_low_center_4_to_5/confusion.json`, `runs/revised/center_4_to_5_level_details.json`
- การตรวจอิสระ `scripts/audit_saved_suite.py` replay raw data และ checkpoint ของ ablation ทั้งสามรันสำเร็จ รวมถึงตรวจ hash ของข้อมูล/model, ตัวเลข accuracy, predicted counts และ final epoch
- สูตร: `configs/full_signedlog_domain.json`, `configs/ablation_full_low_center.json`, `configs/ablation_full_no_center.json`; ตัววิเคราะห์ `scripts/analyze_level_ablation.py`

ตัวอย่างรันคู่ 4→5 แบบปิด center จาก repository root:

```powershell
.\.venv_gpu\Scripts\python.exe -m mpc_cnn.revised --recipe configs\ablation_full_no_center.json --source 4 --target 5 --output runs\revised\ablation_full_no_center_4_to_5 --device cuda --pretrained runs\revised\full_signedlog_domain_resume3\batch4_to_5\pretrained.pt
```
