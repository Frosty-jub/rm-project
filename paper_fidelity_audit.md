# ตรวจความตรงกับ paper: MPC-CNN, Dataset A

> **สถานะหลังโฟลเดอร์เดิมถูกลบ (9 ตุลาคม 2026):** เอกสารนี้เป็นบันทึกการตรวจเมื่อ 29 กันยายน ไม่ใช่ผลตรวจ checkpoint เดิมซ้ำใน workspace ปัจจุบัน ไฟล์ PDF หลัก, `runs/full_figure3`, `runs/full_combined` และ checkpoint เดิมที่อ้างด้านล่างกู้คืนไม่ได้ ตัวเลขเดิมจึงอ้างจากรายงานที่กู้มาเท่านั้น ดูผลฝึกใหม่และการตรวจ checkpoint จริงที่ `WHY_FIRST_RUN_DIFFERS.md` และ `trained_artifact_audit.md` ส่วน Supporting Information, Dataset A และโค้ดที่ตรวจ hash แล้วมีอยู่ใน workspace ใหม่

ตรวจวันที่ 29 กันยายน 2026 จาก PDF หลัก, Supporting Information, โค้ดปัจจุบัน, environment ของแต่ละ run และ checkpoint ที่บันทึกไว้ การตรวจครั้งนี้โหลดโมเดลเพื่อวัดผลและ gradient โดยไม่มี optimizer step หรือการฝึกใหม่

**ข้อสรุป: โครงหลักของโหมด `figure3` สอดคล้องกับรูป แต่ขั้นตอนการฝึกยังไม่ใช่การทำซ้ำ paper ครบถ้วน และยัง reproduce accuracy ไม่สำเร็จ** ผลครบ 18 งานเป็นของโหมด `legacy`; โหมด `figure3` ที่ใช้ค่าหลักครบได้ผลเพียงงาน 1→2 จากทั้งหมด 18 งาน

## รายการเทียบกับต้นฉบับ

| รายการ | หลักฐานใน paper | โค้ดปัจจุบัน / ข้อสรุป |
|---|---|---|
| ข้อมูล | Table 2: 13,910 ตัวอย่าง, 128 features, 10 batches, 6 classes | ใช้ไฟล์ครบ และจัดเป็น 16×8 |
| Backbone | Figure 3: Conv channels 8/16/32/16, kernels 2/2/3/3, padding 1; pool 2; FC 128/64/32 | ตรงกับขนาดและบล็อกที่แสดงในโหมด `figure3`; flatten เป็น 16×9×5=720 |
| Activation | Figure 3 ไม่มี activation ต่อจาก Conv4 และหัวจำแนกระบุ ReLU/FC | โหมดปัจจุบันไม่มี Conv4 activation และอ่าน ReLU/FC เป็น ReLU ก่อน FC; ลำดับใน slash notation ยังเป็นการตีความ |
| โหมดเก่า | เทียบ Figure 3 | `legacy` มี LeakyReLU เพิ่มหลัง Conv4 และไม่มี ReLU ที่หัวจำแนก |
| Formal training | SI Tables S2–S3 | ทั้ง 18 ชุดของ 800 epochs, learning rate, batch size, w ตรงตาราง |
| Pretraining | **Figure 4(a), PDF หน้า 7 / journal หน้า 2912 แสดง Batch 1/2 ไปถึงประมาณ 600 epochs** | `train.py` default เป็น **100**; ไม่ตรง run ที่แสดงในรูป ไม่มีตาราง pretraining แยกทุก task |
| Center loss | Eq. 8 ผลรวมต่อ minibatch; Eq. 12 cross-entropy เฉลี่ยต่อ sample | `model.py` เฉลี่ย center loss เช่นกัน ทำให้สัดส่วน center loss ต่อ CE ต่างจากสมการ |
| Center update | Eq. 10 / Algorithm 1: หารด้วย 1+จำนวนตัวอย่างคลาส | สูตร update ตรง; ค่า beta=0.5 และการเริ่ม center ที่ศูนย์เป็นตัวเลือกของ implementation |
| MMD pretraining | Eq. 11 รวม MMD ทั้งสามระดับ | โครง objective ตรงแนวคิด; biased empirical Gaussian MMD และ median bandwidth เป็นตัวเลือกที่ต้นฉบับไม่ได้ให้โค้ดหรือค่าตายตัว |
| MMD ระหว่าง formal training | ใช้ shared feature extractor เปรียบเทียบ source/target | source ใช้ train mode; target ใช้ eval mode จึงมีผลของ Dropout/BatchNorm ปนในระยะทาง |
| Gradient ผ่าน ensemble weight | Paper ไม่ระบุ stop-gradient | โค้ดใช้ no_grad และ detach ทำให้ task loss ไม่ย้อน gradient ผ่าน MMD weights |
| Ensemble | Eqs. 6–7 reciprocal sigmoid และ weighted sum | โค้ดเพิ่มการ normalize weights; สำหรับ detached weights ที่ใช้อยู่ การ normalize ไม่เปลี่ยน argmax หรือ gradient ของ log-probability loss จึงไม่ใช่คำอธิบายเดี่ยวของช่องว่างนี้ |
| Preprocessing / optimization | z-score, Adam, dynamic LR decay, L2, Dropout | Population/axes ของ z-score, bandwidth, dropout rate, negative slope, L2 coefficient และ decay schedule ไม่ระบุครบ โค้ดเลือก source-only per-feature, median, dropout 0.5, slope 0.01, decay 1e-4 และ cosine |
| Evaluation | ไม่มีรายละเอียดการสร้าง MMD weights ตอนทดสอบ | โค้ดสุ่ม source/target ไม่เกินฝั่งละ 256 ตัวอย่างและใช้ weights คงที่ |

ที่ minibatch ขนาด 90 ในงาน 1→2, center loss แบบ mean มีขนาดเป็น 1/90 ของแบบ sum ตาม Eq. 8 เมื่อใช้ w เดียวกัน นี่เป็นความต่างของ objective ที่ยืนยันได้ แต่การเปลี่ยนเป็น sum เพิ่มแรง penalty และ **ไม่มีหลักฐานว่าจะรักษาการเรียนรู้ที่ยุบอยู่แล้วได้**

## ตรวจผลจาก checkpoint ซ้ำ

รัน `python scripts/audit_reproduction.py` แล้ว target accuracy ของทั้งห้า checkpoint ตรงกับผลเดิม ภายใต้ runtime ที่ตรวจครั้งนี้: Python 3.12.4, PyTorch 2.13.0+cpu, NumPy 1.26.4

| งาน Batch 1→2 | Pretrain epochs | Center loss | Source accuracy | Target accuracy |
|---|---:|---|---:|---:|
| Legacy MPC-CNN | 100 | ใช้ | 100.00% | 44.29% |
| Figure 3 MPC-CNN | 100 | ใช้ | 22.02% | 26.85% |
| Figure 3 ตัด pretraining | 0 | ใช้ | 74.61% | 36.98% |
| Figure 3 ตัด center loss | 100 | ไม่ใช้ | 100.00% | 42.85% |
| Figure 3 CNN | 0 | ไม่ใช้ | 100.00% | 53.78% |

Paper Tables 4–5 รายงาน MPC-CNN 85.06% และ CNN 70.74% สำหรับงานนี้ ส่วนค่าเฉลี่ยครบ 18 งานใน `runs/full_combined` เป็น **legacy**: long-term 36.75% เทียบ 76.06%, short-term 49.13% เทียบ 82.07% ไม่ใช่ค่าเฉลี่ยของโหมด Figure 3 ปัจจุบัน

## กลไกที่พบในโมเดลที่ผลต่ำ

ใน `runs/full_figure3/long_1_to_2_mpc.pt` เมื่อประเมิน source ทั้ง 445 ตัวอย่าง:

- หลัง ReLU ก่อนหัวจำแนกระดับ 1 มี vector เป็นศูนย์ทุกมิติ 440 จาก 445 ตัวอย่าง
- ระดับ 2 และ 3 มี vector เป็นศูนย์ทุกมิติ **445 จาก 445 ตัวอย่าง** ทั้งสองระดับ
- Task-loss gradient ต่อ feature ระดับ 2 และ 3 เท่ากับศูนย์ใน eval probe นี้ หัวเหล่านี้จึงให้ logits ที่ขึ้นกับ bias อย่างเดียวบน source
- โมเดลทำนาย raw class 2 ทุกตัวอย่าง ทั้ง source 445 ตัวและ target 1,244 ตัว ค่า 22.02% และ 26.85% ตรงกับสัดส่วน class 2 ในแต่ละ batch

นี่เป็นหลักฐานตรงว่าตัวจำแนกสูญเสียความสามารถในการแยกคลาสใน checkpoint นี้ ไม่ใช่แค่โมเดลที่เรียนรู้ source ได้ดีแล้วเจอ drift ใน target อย่างเดียว ส่วนสาเหตุที่การฝึกมาถึงจุดนี้ยังไม่ถูกแยกครบ การตัด center loss ทำให้ source กลับเป็น 100% เป็นหลักฐานว่าการตั้งค่าและปฏิสัมพันธ์กับ pretraining/activation มีส่วน แต่ไม่ได้พิสูจน์ว่า center loss หรือ pretraining ใน paper ผิด

อีก probe ใช้ tensor เดียวกันเป็นทั้ง source และ target ของโมเดลใหม่ เมื่อ source ใช้ train mode แต่ target ใช้ eval mode ได้ MMD โดยประมาณ `[0.735, 0.722, 0.671]`; เมื่อทั้งสองใช้ eval mode ได้ `[1e-5, 1e-5, 1e-5]` ซึ่งเป็น numerical floor ของโค้ด ดังนั้น MMD ที่ใช้ formal training มีระยะทางที่เกิดจากโหมดของเครือข่ายปนอยู่จริง ผลกระทบต่อ accuracy ยังไม่ได้วัดด้วยการฝึกแบบควบคุม

## ข้อจำกัดและสิ่งที่แก้จากคำอธิบายเดิม

1. คำกล่าวเดิมว่า paper ไม่ให้จำนวน pretraining epochs เลยไม่ถูกต้อง: Figure 4(a) ให้หลักฐานประมาณ 600 สำหรับ Batch 1/2 แต่ไม่ได้แจกแจงทุก task
2. Center-loss mean ไม่ควรถูกจัดเป็นเพียงรายละเอียดที่ไม่ระบุ เพราะ Eq. 8 เขียนผลรวมชัดเจน
3. `environment.json` แสดงว่า legacy runs ใช้ Python 3.12.13 / PyTorch 2.14.0+cpu / NumPy 2.3.5 ขณะที่ Figure 3 และ diagnostic runs ใช้ Python 3.12.4 / PyTorch 2.13.0+cpu / NumPy 1.26.4 การเทียบ legacy กับ Figure 3 จึงไม่ได้เปลี่ยนเฉพาะ activation
4. การตัด pretraining เปลี่ยนลำดับ RNG และการเปรียบเทียบ CNN กับ MPC ยังต่าง initialization (CNN ใช้ Kaiming-normal) จึงเป็น diagnostic จาก seed เดียว ไม่ใช่ข้อพิสูจน์เหตุและผลครบถ้วน
5. Paper กล่าวถึง final accuracy แต่ไม่ได้ระบุ best-checkpoint policy, seed หรือจำนวนรันซ้ำ จึงไม่มีหลักฐานให้กล่าวว่าผู้เขียนเลือก best target result
6. การให้ target อัปเดต BatchNorm running statistics ไม่ใช่ความผิดโดยนิยาม เพราะ paper ไม่ระบุ policy; การเปลี่ยนมา eval target ในโค้ดนี้ช่วยรักษา source running stats แต่สร้างปัญหา mode mismatch ที่วัดพบข้างต้น
7. Tests 5 รายการผ่าน แต่ตรวจ data schema, finite gradients และพฤติกรรมที่กำหนดไว้ ไม่ได้ยืนยันความตรงกับ paper หรือว่าการฝึกเรียนรู้สำเร็จ
8. ไม่พบลิงก์โค้ดผู้เขียนใน PDF/SI ที่ตรวจ ไม่มีหลักฐานเพียงพอให้กล่าวว่าไม่มีโค้ดเผยแพร่ที่อื่น

ก่อนเรียกเป็นผล reproduction ต้องทำให้ loss และ MMD computation สอดคล้องกัน บันทึกทุกตัวเลือกให้ครบ แล้วทดลอง Batch 1→2 ด้วยงบ pretraining ตาม Figure 4(a) ภายใต้ environment เดียวกัน พร้อมตรวจ source learning ก่อนรัน 18 งาน การเปลี่ยนค่าที่ paper ไม่ระบุควรเป็น sensitivity analysis และไม่เลือกจากคะแนน target เพื่ออ้างว่าเป็นค่าของ paper

## หลักฐานที่ตรวจซ้ำได้

- [โค้ดตรวจ checkpoint](scripts/audit_reproduction.py)
- [ผลตรวจ feature, gradient, accuracy, environment และ MMD mode probe](runs/audit_20260929/checkpoint_review.json)
- [รายงานผลรันเดิม](reproduction_results.md)
- [บทความ ACS](https://doi.org/10.1021/acssensors.4c03655), Figure 3–4, Eqs. 3–12, Tables 2–5
- [Supporting Information](supporting_information.pdf), Tables S2–S3
