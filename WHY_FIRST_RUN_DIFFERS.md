# ทำไมผลรัน MPC-CNN รอบแรกจึงต่างจาก paper และแก้อย่างไร

อัปเดต 10 ตุลาคม 2026 — Dataset A, seed 42. ตัวเลขของรอบแรกต่อไปนี้มาจาก `reproduction_results.md` ที่กู้จากบันทึกงานเดิม; ไฟล์ checkpoint และผล CSV ของรอบนั้นถูกลบไปพร้อมโฟลเดอร์และไม่พบใน Recycle Bin จึงตรวจไฟล์นั้นซ้ำไม่ได้ โค้ดที่สร้าง checkpoint รุ่นใหม่กู้กลับมาได้โดย SHA-256 ตรงกับ snapshot ที่ GPT-6 Astra เคยตรวจแล้ว ชุดข้อมูลกู้จาก [UCI dataset 224](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) และแปลง line ending ให้ hash ของ batch 1/2 ตรงกับของเดิม

## ผลเดิมและขอบเขตของคำว่า “ตาม paper”

| การทดลองเดิม | ของเรา | Paper | ต่างกัน |
|---|---:|---:|---:|
| Long-term 9 งาน, legacy MPC-CNN | 36.75% | 76.06% | −39.30 จุด |
| Short-term 9 งาน, legacy MPC-CNN | 49.13% | 82.07% | −32.94 จุด |
| Batch 1→2, Figure 3 layout | 26.85% | 85.06% | −58.21 จุด |

`legacy` คือโครงสร้างรอบแรก มี LeakyReLU เพิ่มหลัง convolution ชั้นที่สี่ และไม่มี ReLU ก่อนหัวจำแนกตามการอ่าน Figure 3 ของเรา ตัวเลข 36.75%/49.13% เป็นของ `legacy` ครบ 18 protocol rows ส่วน 26.85% เป็นโมเดลแก้ layout แต่รันเพียง 1→2; ห้ามเอามาปนเป็น macro ของรุ่นเดียวกัน

หลังการกู้ รัน `legacy` 1→2 ใหม่ด้วย seed 42, pretraining 100, formal 800 และ PyTorch 2.7.1+cu126 บน RTX 2060 ได้ **47.27%** (`runs/recovered_legacy_1_to_2/results.csv`) ต่างจากค่ารายงานเก่าของงานเดียวกัน 44.29% อยู่ 2.98 จุด และยังห่าง paper 37.79 จุด รันนี้เป็นการทำซ้ำในสภาพแวดล้อมใหม่ ไม่ใช่การกู้ checkpoint เดิม จึงยืนยันได้ว่าช่องว่างใหญ่ยังเกิดซ้ำ แต่แยกผลของเวอร์ชันไลบรารี/ฮาร์ดแวร์กับความไม่แน่นอนของการฝึกไม่ได้จากการรันครั้งเดียว

## สาเหตุที่ตรวจพบจริง

| หลักฐาน | ทำไมกระทบผล | วิธีแก้ / สถานะ |
|---|---|---|
| Figure 3 checkpoint เดิมทำนายทุกตัวอย่างเป็นคลาสเดียว ได้ source 22.02% และ target 26.85%; feature หลัง ReLU ของหัวที่ 2–3 เป็นศูนย์บน source ทั้ง 445 ตัวอย่าง | โมเดลไม่เรียนรู้แยกคลาสบน source แล้ว จึงไม่มีทางโอนย้ายได้ดี | Runner ใหม่ทำให้ชนิด activation ของหัวเป็นค่าเลือกที่บันทึกไว้; pilot ใช้หัว linear และตรวจ source accuracy, feature fractions และจำนวนคลาสที่ทำนายทุกงาน |
| โค้ดเดิมคำนวณ MMD ของ source ใน train mode แต่ target ใน eval mode; ป้อนข้อมูลเดียวกันสองฝั่งยังได้ MMD ประมาณ 0.67–0.73 | BN/Dropout สร้าง “ความต่างโดเมน” เทียมในน้ำหนัก ensemble | Runner ใหม่คำนวณทั้งสองฝั่งในโหมดเดียวกัน; identical-domain MMD ใกล้ศูนย์ใน unit test |
| Loss ที่โค้ดเดิมใช้เฉลี่ย center loss แต่ Eq. 8 ของ paper เขียนผลรวมต่อ minibatch; Eq. 12 ใช้ค่าเฉลี่ย CE | อัตราส่วนของ center penalty ต่อ CE ต่างจากสมการ เช่น batch 90 ต่าง 90 เท่าเมื่อ coefficient เท่ากัน | Runner ใหม่มี `center_reduction=sum/mean` ชัดเจน และบันทึก multiplier/warmup; สูตร `equation_sensitivity` ใช้ sum ส่วนสูตรปรับใช้ mean พร้อมระบุว่าเบี่ยงจาก paper |
| Pretraining เดิม 100 epochs; Figure 4(a) ของ paper แสดงตัวอย่าง Batch 1/2 ถึงราว 600 epochs | งบฝึกของตัวอย่างนี้ไม่ตรงกับรูป; แต่ paper ไม่แจกงบ pretraining ทุกคู่ | มี recipe 600 epochs สำหรับรันเต็ม และระบุ pilot 50 epochs เป็นการคัดกรองเท่านั้น |
| เมื่อตัด pretraining ใน Figure 3 run เดิม source accuracy เพิ่มจาก 22.02% เป็น 74.61%; เมื่อตัด center loss เพิ่มเป็น 100% | ชี้ว่าการตั้งค่า pretraining/center ที่เราเลือกมีปฏิสัมพันธ์กับอาการยุบ | Runner ใหม่ให้เลือก CE+MMD pretraining, center warmup, ค่า center ที่เล็กลง; ต้องทำ ablation แบบคุมค่าอื่นเพื่อบอกผลของแต่ละจุด |

ผลทดลองข้างบนพิสูจน์ว่า *implementation เดิมมีปัญหาและไม่ตรงสมการ/รูปบางจุด* แต่ยังไม่พิสูจน์ว่าเปลี่ยนจุดใดจุดเดียวแล้วจะได้คะแนน paper

อีกประเด็นในรายงานเก่าคือเคยอนุมานชื่อ gas ของ label 1–6 จากจำนวนตัวอย่างในตาราง paper ซึ่งขัดกับคำอธิบายรหัสคลาสของ UCI Dataset 224 การฝึกและประเมินใช้ label ตัวเลขจากไฟล์เดียวกันทุก batch โดยตรง จึงไม่ใช่สาเหตุของช่องว่าง accuracy; ผมแก้ `reproduction_note.md` ให้ยึด mapping ที่ UCI ระบุและไม่ตีความชื่อ gas จากจำนวนตัวอย่างอีก

หลังการกู้ ผมรัน [identical-domain MMD probe](runs/audit_recovered/identical_domain_mmd.json) ซ้ำกับโค้ดเดิมและข้อมูลสุ่ม 32 ตัวอย่าง: source/target เป็น input ชุดเดียวกัน ได้ MMD `[0.7353, 0.7217, 0.6710]` เมื่อใช้คนละโหมด แต่ได้ประมาณ `0.00001` ทั้งสามระดับเมื่อใช้ eval mode เหมือนกัน จึงยืนยันกลไกนี้ได้โดยไม่ต้องพึ่ง checkpoint ที่หายไป

## รายละเอียดที่ paper ยังไม่กำหนดเป็นโค้ดตายตัว

ลำดับที่แน่นอนของหัว ReLU/FC, กลุ่มตัวอย่างสำหรับ z-score, Gaussian bandwidth ของ MMD, dropout, ค่า L2, การเริ่มต้นและอัปเดต class centers, LR decay, gradient ผ่านน้ำหนัก ensemble และการสุ่มตัวอย่างเพื่อคำนวณน้ำหนักตอนประเมิน ยังต้องใช้สมมติฐานที่บันทึกใน config ของแต่ละรัน

การตรวจ gradient อิสระโดย Astra พบว่า median bandwidth ที่ detach ให้ gradient ผลัก feature ให้หด แม้ค่า MMD ที่คำนวณใหม่จะเกือบคงที่เมื่อขยาย/หด feature พร้อมกัน นี่เป็นกลไกที่เป็นไปได้ ไม่ใช่หลักฐานว่ามันเป็นสาเหตุหลักของ checkpoint เดิม Runner ใหม่ทดลอง median แบบ differentiable และแบบ fixed ได้

## สิ่งที่เปลี่ยนเพื่อหาผลใกล้ paper

ทดลอง preprocessing สามแบบที่กำหนดไว้ก่อน: domain z-score, signed-log ตามด้วย domain z-score, และ source z-score ตามด้วย row L2 ใช้ source labels และ target features ที่ไม่ติด label เพื่อฝึก; target labels ใช้คำนวณ accuracy อย่างเดียว การคัดสูตรจาก accuracy ของ target ยังเป็น **target-informed exploratory selection** แม้ไม่ใช้ target labels ใน gradient ดังนั้นผลที่ดีขึ้นจะเรียกว่า MPC-CNN ที่ปรับจาก paper ไม่ใช่ reproduction ของโค้ดผู้เขียน

ผล SVC diagnostic ที่รันซ้ำหลังการกู้เป็นอีกโมเดลหนึ่ง: signed-log+domain z-score ได้ long/short macro 64.20%/72.66%; source z-score+row L2 ได้ 53.12%/71.97%; signed-log+pooled z-score ได้ 46.06%/79.59% คะแนนเหล่านี้ใช้ชี้ preprocessing ที่ควรลองใน CNN เท่านั้น ห้ามนำไปอ้างว่า MPC-CNN ได้คะแนนเท่านี้

## ผล MPC-CNN ใหม่

Pilot รันจริงบน RTX 2060 ด้วย pretraining 50 epochs และ formal training 150 epochs ทุกสูตรมี source accuracy 100% ทั้งสามคู่ ตัวเลขต่อไปนี้เป็น target accuracy จาก epoch สุดท้าย ไม่ใช่การเลือก checkpoint จาก target labels:

| สูตร | 1→2 | 1→6 | 4→5 | เฉลี่ยสามคู่ |
|---|---:|---:|---:|---:|
| domain z-score | 69.05% | 91.83% | 69.04% | 76.64% |
| signed-log + domain z-score | 75.00% | 92.35% | 83.76% | 83.70% |
| source z-score + row L2 | 71.54% | 83.61% | 95.94% | 83.70% |
| paper, เฉพาะสามคู่เดียวกัน | 85.06% | 95.16% | 94.71% | 91.64% |

ไฟล์ตรวจสอบได้ที่ `runs/revised/pilot_domain/tasks.csv`, `runs/revised/pilot_signedlog_domain/tasks.csv` และ `runs/revised/pilot_source_row_l2/tasks.csv` พร้อม `summary.json`, `result.json`, `model.pt` ของแต่ละคู่ ผล signed-log สูงกว่า row L2 เพียง 0.0043 จุดเปอร์เซ็นต์ จึงแทบเสมอกัน กติกาที่บันทึกไว้ก่อนรันเลือก signed-log เพื่อรันเต็ม; ความต่างระดับนี้ไม่ใช่หลักฐานว่าสูตร signed-log ดีกว่าโดยทั่วไป ผล row L2 ใน 4→5 แสดงว่า preprocessing ยังเป็นแหล่งความไวของคะแนน

สูตรที่เลือกไว้ใน `configs/full_signedlog_domain.json` เพิ่มงบฝึกเป็น 600+800 epochs โดยคงค่าอื่นจาก pilot คู่ 1→2 จบที่ **73.55% target / 100% source** เทียบ paper 85.06% ยังห่าง **11.51 จุด** และต่ำกว่า pilot 75.00% อยู่ 1.45 จุด การเพิ่ม epochs อย่างเดียวจึงไม่ได้ทำให้คู่นี้ดีขึ้น ชุดเต็ม [ตารางรายคู่และ macro](runs/revised/full_signedlog_domain_resume3/comparison.md) รันครบ 17 คู่แล้ว: long-term **65.26%** เทียบ paper **76.06%** (−10.79 จุด) และ short-term **66.07%** เทียบ **82.07%** (−16.00 จุด) โหลด checkpoint ประเมินซ้ำและตรวจ hash ผ่าน **17/17** ใน `runs/revised/full_signedlog_domain_resume3/audit.json` สูตรนี้จงใจปรับ preprocessing, head, pretraining และ center loss จาก paper จึงตีความเป็น exploratory modified MPC-CNN เท่านั้น

สูตรเดียวกันที่ใช้งบสั้น 50+150 epochs รันครบ **17 คู่ไม่ซ้ำ** แล้ว: [ตารางรายคู่และ macro](runs/revised/pilot_signedlog_full_suite/comparison.md) ได้ long-term **66.85%** เทียบ paper **76.06%** (−9.20 จุด), short-term **70.06%** เทียบ **82.07%** (−12.01 จุด), source accuracy 100% ทุกคู่ โหลด checkpoint ประเมินซ้ำจากข้อมูลดิบและตรวจ hash ผ่าน **17/17** ใน `runs/revised/pilot_signedlog_full_suite/audit.json` ผลนี้ดีขึ้นจาก legacy macro เก่า 36.75%/49.13% แต่ยังไม่ reproduce paper และงบฝึกไม่ตรงตัวอย่าง Figure 4(a) การตัดสินใจรันงบสั้นครบ protocol เกิดหลังเห็นผล target ของงบเต็มบางคู่ จึงเป็น target-informed exploratory เพิ่มอีกชั้น

สูตร source z-score + row L2 ที่ใช้งบสั้น 50+150 epochs รันครบ **17 คู่ไม่ซ้ำ** เช่นกัน: [ตารางรายคู่และ macro](runs/revised/pilot_row_l2_full_suite_resume/comparison.md) ได้ long-term **62.72%** เทียบ paper **76.06%** (−13.33 จุด) และ short-term **76.69%** เทียบ **82.07%** (−5.38 จุด), source accuracy 100% ทุกคู่ ตัวตรวจอิสระโหลด checkpoint จากข้อมูลดิบและตรวจ hash ผ่าน **17/17** ใน `runs/revised/pilot_row_l2_full_suite_resume/audit.json` สูตรนี้ทำ 4→5 **95.94%** เทียบ paper **94.71%**, 7→8 **91.50%** เทียบ **90.20%** แต่ 3→4 ได้เพียง **64.60%** เทียบ **96.13%** จึงห้ามเลือกเฉพาะคู่ที่ดีมาอ้างว่า reproduce paper การรัน row L2 ครบ protocol เกิดหลังเห็นคะแนน target ของ pilot และของ signed-log บางคู่ จึงเป็นการประเมิน exploratory ที่ target-informed ไม่ใช่ผลทดสอบอิสระ

| สูตรเดียวตลอด 17 คู่ | Pretrain + formal | Long-term | Short-term |
|---|---:|---:|---:|
| signed-log + domain z-score | 50 + 150 | 66.85% | 70.06% |
| source z-score + row L2 | 50 + 150 | 62.72% | 76.69% |
| signed-log + domain z-score | 600 + 800 | 65.26% | 66.07% |
| paper | ตามบทความ/SI | 76.06% | 82.07% |

ผลที่ใกล้ paper ที่สุดใน **สูตรเดียวครบทั้ง protocol** คือ long-term 66.85% (ยังห่าง 9.20 จุด) และ short-term 76.69% (ยังห่าง 5.38 จุด) ซึ่งมาจากคนละสูตรและเลือกเปรียบเทียบหลังเห็น target labels จึงห้ามรวมเป็นผลงานจากโมเดลเดียว การเลือกสูตรที่ชนะเป็นรายคู่จาก target labels จะได้ short-term 80.11% เชิงคณิตศาสตร์ แต่เป็นเพียงเพดานย้อนหลังที่ใช้ข้อมูลทดสอบ ไม่ใช่ผลการประเมินที่ใช้ได้หรือวิธีแก้การโอนย้ายโดยไม่มีป้ายกำกับ

สูตร `equation_sensitivity` ที่อ่านสมการแบบหนึ่งอย่างตรงตัวกว่า ใช้ MMD-only pretraining, ReLU head, center แบบ sum/น้ำหนักตาม SI, center เริ่มศูนย์ และ raw differentiable ensemble weights จบ 1→2 ที่ **26.85% target / 22.02% source** โดยทำนายยุบ ไม่ได้แปลว่า paper ใช้โค้ดนี้ เพราะ Eq. 7/12 ยังมีความกำกวมเรื่องการ normalize weights และตัวเลือกอื่นไม่ได้ระบุครบ ผลนี้เปลี่ยนหลายปัจจัยพร้อมกัน จึงยังแยกไม่ได้ว่าข้อใดทำให้ยุบ

การทดลองควบคุม `configs/full_signedlog_center_sum.json` เปลี่ยน **เฉพาะ** center reduction จาก mean เป็น sum และใช้ pretraining checkpoint เดียวกันกับสูตรเต็ม 1→2 (`pretrained_checkpoint_sha256` บันทึกในผล) seed และค่าอื่นเท่ากัน ที่ epoch 800 ได้ **54.66% target / 100% source** เทียบสูตร mean **73.55% / 100%** ลดลง **18.89 จุด** ตรวจซ้ำจากโมเดลที่บันทึกผ่าน `scripts/audit_saved_suite.py` แล้ว ผลนี้ระบุผลเชิงสาเหตุของการเปลี่ยน reduction ภายใต้สูตรทดลองหนึ่งและคู่ 1→2 เท่านั้น ไม่พิสูจน์ว่า center sum เป็นสาเหตุเดียวของอาการยุบในสูตร equation sensitivity หรือว่า paper ใช้ mean

การทดลองควบคุม `configs/full_signedlog_mixed_mmd.json` เปลี่ยน **เฉพาะ**วิธีคำนวณน้ำหนัก MMD ระหว่าง formal training เป็น source train-mode/target eval-mode แบบเดิม โดยใช้ pretraining checkpoint เดียวกัน ได้ **73.47% target / 100% source** เทียบโหมดเดียวกันสองฝั่ง **73.55% / 100%** ต่างเพียง **−0.08 จุด** ในคู่นี้ ข้อผิดพลาดเรื่อง MMD เทียมจึงเป็นปัญหาความถูกต้องของการคำนวณที่ตรวจพบ แต่จากการทดลองควบคุมนี้ยังไม่มีหลักฐานว่าเป็นตัวการหลักของช่องว่าง accuracy

น้ำหนัก ensemble ตอนประเมินของสองรันใกล้กันมาก: โหมดเดียวกัน `[0.331634, 0.334100, 0.334266]`; โหมดผสม `[0.331640, 0.334095, 0.334265]` จึงสอดคล้องกับผล accuracy ที่แทบไม่เปลี่ยน ข้อสังเกตนี้เป็นการตีความผลของสูตรที่ normalize น้ำหนักและ detach gradient เท่านั้น ไม่ครอบคลุมทุกวิธีให้น้ำหนักใน paper

ความผิดพลาดของแต่ละคู่ไม่เหมือนกัน: checkpoint `equation_sensitivity` 1→2 ทำนายคลาส index 1 (นับจากศูนย์) **1,243 จาก 1,244** target ตัวอย่าง จึงเป็นการยุบเกือบทั้ง batch ส่วนสูตรเต็ม 1→4 ฝึก source ได้ 100% แต่ target เหลือ 64.60%; [confusion matrix](runs/revised/full_signedlog_domain_resume/batch1_to_4/confusion.json) พบว่า class index 2 ถูกทำนายผิด 12/12 และ index 3 ถูก 3/30 ขณะที่ index 1 และ 4 ถูกทั้งหมด จึงเป็นความล้มเหลวการโอนย้ายเฉพาะคลาส ไม่ใช่โมเดลทำนายคลาสเดียวแบบรันที่ยุบ

สำหรับสูตรที่ปรับแล้ว 1→2 ที่ได้ 73.55%, [confusion matrix](runs/revised/full_signedlog_1_to_2/batch1_to_2/confusion.json) แสดงว่ามีการทำนายครบทั้งหกคลาส แต่ recall ของ target class index 0 เพียง **3/164** และ index 3 เท่ากับ **0/109** จึงแก้อาการยุบทั้งโมเดลได้แล้ว แต่ยังแก้การจำแนกบางคลาสข้าม batch ไม่สำเร็จ ความผิดพลาดเฉพาะคลาสนี้เป็นสิ่งที่วัดได้ ไม่ใช่หลักฐานว่าปัจจัยใดใน paper ทำให้เกิดมัน

ค่าตัวเลขใน paper บางงาน **ไม่สามารถเท่ากันสองตำแหน่งจากการทดสอบหนึ่งครั้งบน target batch เต็มชุด** ได้ในทางคณิตศาสตร์ เช่น target Batch 4 มี 161 ตัวอย่าง; 154/161 = 95.65%, 155/161 = 96.27% แต่ paper รายงาน 3→4 ที่ 96.13% จึงไม่มีจำนวนตัวอย่างถูกต้องที่เป็นจำนวนเต็มซึ่งให้ 96.13% บน 161 ตัวอย่างเดียวกัน นี่พิสูจน์เพียงว่าเกณฑ์รวมผล/ชุดประเมินของตัวเลขนั้นไม่ใช่ single-pass บน target เต็มชุดแบบ runner นี้ อาจเป็นค่าเฉลี่ยหลายรันหรือเลือก subset แต่เอกสารที่ตรวจยังไม่ยืนยันว่าแบบใด ความต่างข้อนี้ไม่ได้อธิบายช่องว่างใหญ่ 17 จุดของเรา

การฝึกนานขึ้นไม่ได้ช่วยทุกคู่: signed-log pilot 4→5 ที่ 50+150 epochs ได้ **83.76%**, แต่สูตรเดียวกันที่ 600+800 epochs ได้ **56.85%** (source ยัง 100%) [confusion matrix](runs/revised/full_signedlog_short_worker_resume/batch4_to_5/confusion.json) แสดงว่า target class index 4 ถูกผิด 63/63 ตัวอย่าง ผลนี้เป็นหลักฐานว่า checkpoint สุดท้ายของงบฝึกเต็มเกิดการโอนย้ายแย่ลงภายใต้สูตรที่เลือก จึงควรทดลองงบสั้นครบ protocol แยกเป็น sensitivity โดยระบุว่าไม่ใช่งบตัวอย่าง Figure 4(a) ของ paper
ใน [pilot confusion](runs/revised/pilot_signedlog_domain/batch4_to_5/confusion.json) คลาสเดียวกันยังถูก **33/63** และ class index 3 ถูก **46/46**; รันเต็มเหลือ **0/63** และ **26/46** ตามลำดับ นี่ชี้ว่าคะแนนที่ลดลงเกิดจากความผิดพลาดข้ามคลาสจริง ไม่ใช่ความต่างในสูตรคำนวณ accuracy

แหล่งสมการและผลอ้างอิง: [บทความ ACS Sensors](https://doi.org/10.1021/acssensors.4c03655), [Supporting Information](https://acs.figshare.com/articles/journal_contribution/Gas_Sensor_Drift_Compensation_Using_Semi-Supervised_Ensemble_Classifiers_with_Multi-Level_Features_and_Center_Loss/28752293), [UCI Dataset A](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset), `paper_fidelity_audit.md`, `astra_review.md`

## ผล ablation เพิ่มเติม 10 ตุลาคม 2026

รันคู่ 4→5 ใหม่แบบเปลี่ยนเฉพาะแรงของ center loss หลังโหลด pretraining checkpoint เดียวกัน: ค่าเดิม 56.85%, ลด center 10 เท่า 67.51%, ปิด center 80.20%; source ยัง 100% ทุกกรณี Target class index 4 ที่เคยถูก 0/63 กลับถูก 27/63 เมื่อปิด center ขณะที่คู่ 1→2 ปิด center แล้วลดจาก 73.55% เป็น 70.90% จึงเป็นปัญหาที่ขึ้นกับคู่ข้อมูลและแรง regularization ไม่ใช่ข้อสรุปให้ลบ center loss ออกจากทุกงาน รายละเอียดหลักฐาน การตรวจ checkpoint และข้อจำกัดอยู่ใน [รายงาน ablation](ACCURACY_ABLATION_2026-10-10.md)
