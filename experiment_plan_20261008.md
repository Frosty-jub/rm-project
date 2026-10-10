# แผนทดลอง MPC-CNN หลังการกู้ไฟล์ (8 ตุลาคม 2026)

## ที่มาและสถานะ

โฟลเดอร์เดิมถูกลบก่อนจบรันตัวแทน โค้ด `mpc_cnn/revised.py`, `model.py`, `data.py`, `train.py`, ตัวรัน suite และ tests กู้จากบันทึก patch ในเครื่องแล้ว SHA-256 ตรงกับ snapshot ที่ GPT-6 Astra ตรวจเมื่อ 29 กันยายน 2026 ชุดข้อมูลกู้จาก UCI dataset 224 แล้วแปลงบรรทัด LF เป็น CRLF ให้เหมือน snapshot เดิม: `batch1.dat` และ `batch2.dat` มี SHA-256 ตรงกับค่าใน checkpoint ที่บันทึกไว้ ผลรันเก่าและ checkpoint เก่าที่อยู่ในโฟลเดอร์ `runs/` ไม่ได้อยู่ใน Recycle Bin และกู้กลับมาเป็นไฟล์ไม่ได้ ตัวเลขเก่าใน `reproduction_results.md` เป็นรายงานที่กู้จากบันทึก patch เท่านั้น

## การทดลองที่กำหนดก่อนดูผลใหม่

ทดสอบสามสูตรบนคู่ 1→2, 1→6, 4→5 ที่ seed 42: `pilot_domain`, `pilot_signedlog_domain`, และ `pilot_source_row_l2` แต่ละสูตรใช้ CE+MMD pretraining 50 epochs และ formal training 150 epochs ตาม config ของงาน สูตรทดลองเหล่านี้ใช้หัวจำแนกเชิงเส้น, center loss แบบ mean และ coefficient 0.01×ค่าตารางเสริม, source-mean centers และ warmup, จึงเป็น **MPC-CNN ที่ปรับจาก paper** ไม่ใช่การทำซ้ำสูตรตามสมการทั้งหมด ใช้ label ของ target เพียงเพื่อประเมินผล ไม่ให้ gradient, center, สถิติ preprocessing หรือเลือก checkpoint อย่างไรก็ตามการเลือกสูตรจาก target accuracy เป็นการเลือกที่อาศัย target labels จึงต้องรายงานเป็น exploratory target-informed

กติกาคัดสูตรจาก pilot: สูตรที่ source accuracy อย่างน้อย 95% ทั้งสามคู่จึงผ่านเกณฑ์การเรียนรู้ source; ในกลุ่มนั้นเลือกค่าเฉลี่ย target accuracy แบบให้สามคู่มีน้ำหนักเท่ากันสูงสุด หากไม่มีสูตรใดผ่านเกณฑ์ ให้เลือกสูตรที่มี source macro สูงสุดก่อน ผลจากสามคู่เป็นการคัดสูตรแบบ target-informed อยู่ดี

หากพบสูตรที่ฝึก source ได้และ loss มีค่าปกติ จะตรึงสูตรนั้นแล้วรัน protocol ทั้ง 17 คู่ไม่ซ้ำ (9 long-term และ 9 short-term โดย 1→2 นับในทั้งสอง macro) ด้วย 600 pretraining และ 800 formal epochs แล้วรายงานช่องว่างจาก 76.06%/82.07% ใน paper ทุก checkpoint เลือกจาก epoch สุดท้าย

ทดสอบ `equation_sensitivity` บน 1→2 แยกต่างหากด้วย 600+800 epochs, MMD-only pretraining, center loss แบบ sum, และ raw differentiable ensemble weights เพื่อศึกษาการอ่านสมการหนึ่งแบบ Eq. 7 ไม่ normalize น้ำหนัก ขณะที่ Eq. 12 เรียกผลลัพธ์ว่า probability จึงยังถือเป็นข้อสมมติ ไม่ใช่โค้ดต้นฉบับที่ยืนยันแล้ว

ผล SVC screen ก่อนลบเคยดู target scores ของทุก task แล้ว ดังนั้นแม้จะตรึงสูตรก่อน suite เต็ม ผล suite ก็ไม่ใช่การประเมินที่ไม่เคยเห็น target มาก่อน ต้องใช้คำว่า exploratory ตลอด

## การตัดสินใจเพิ่มเติมหลังเห็นผลบางส่วน (9 ตุลาคม 2026)

สูตร signed-log 600+800 ได้ 4→5 เพียง 56.85% ขณะที่ pilot 50+150 สูตรเดียวกันได้ 83.76% และ source accuracy ของทั้งคู่ 100% จึงจะรัน **สูตร pilot 50+150 ครบ 17 คู่** เป็น sensitivity ของงบฝึก โดยไม่เปลี่ยนสูตรระหว่างคู่และใช้ checkpoint epoch สุดท้ายเหมือนเดิม การตัดสินใจนี้เกิดหลังเห็นคะแนน target บางส่วนและไม่ได้เป็นส่วนของกติกาคัดสูตรก่อนทดลอง ต้องรายงานเป็น exploratory target-informed เพิ่มอีกชั้น ห้ามอ้างว่าเป็นงบฝึกที่ paper กำหนดทุกคู่

หลังสูตรงบสั้น signed-log จบครบ 17 คู่ ได้ long 66.85% / short 70.06% ยังต่ำกว่า paper 76.06% / 82.07% และ pilot `source z-score + row L2` เคยได้เฉลี่ยสามคู่แทบเสมอกับ signed-log (ต่าง 0.0043 จุด) โดย 4→5 ได้ 95.94% จึงจะรัน **row-L2 50+150 ครบ 17 คู่** ด้วย recipe ที่ล็อกไว้เดิมเพื่อเทียบ fixed preprocessing recipes ทั้งชุด การทดลองนี้ตัดสินใจหลังเห็น target outcomes มากขึ้นและยังเป็น target-informed exploratory ห้ามเลือก preprocessing ที่ดีที่สุดเป็นรายคู่เพื่ออ้างว่าเป็น paper reproduction
