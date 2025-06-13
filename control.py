import serial
import time
import subprocess
import os
from datetime import datetime
import cv2
import numpy as np

# Viteza, m1, m2, m4, m3, m5, m6
# m1=baza, m2=brat, m3=cot, m4=inchietura, m5=rotatie gripper, m6=gripper
PozInitiala = "30,90,140,90,130,90,10\n"

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2) 

def Poz1():
    comanda = "30,85,150,115,180,90,10\n"
    ser.write(comanda.encode())
    time.sleep(2)
    comanda = "30,85,150,115,180,90,70\n"
    ser.write(comanda.encode())
    time.sleep(10)
    ser.write(PozInitiala.encode())
    time.sleep(2) 

def Poz2():
    comanda = "30,115,130,130,180,90,10\n"
    ser.write(comanda.encode())
    time.sleep(2)
    comanda = "30,115,130,130,180,90,70\n"
    ser.write(comanda.encode())
    time.sleep(10)
    ser.write(PozInitiala.encode())
    time.sleep(2) 

def Poz3():
    comanda = "30,50,140,130,170,90,10\n"
    ser.write(comanda.encode())
    time.sleep(2)
    comanda = "30,50,140,130,170,90,70\n"
    ser.write(comanda.encode())
    time.sleep(10)
    ser.write(PozInitiala.encode())
    time.sleep(2) 

def Poz4():
    comanda = "30,45,110,160,170,90,10\n"
    ser.write(comanda.encode())
    time.sleep(2)
    comanda = "30,45,110,160,170,90,70\n"
    ser.write(comanda.encode())
    time.sleep(10)
    ser.write(PozInitiala.encode())
    time.sleep(2) 

def face_poza(nume=None):
    if nume is None:
        nume = f"poza_{datetime.now().strftime('%H%M%S')}.png"
    subprocess.run(["libcamera-still", "-o", nume, "--encoding", "png", "--timeout", "2000"])
    return nume

def detecteaza_puncte_albastre(cale_imagine, nume_output=None):
    try:
        img = cv2.imread(cale_imagine)
        if img is None:
            return []
        
        height, width = img.shape[:2]
        padding_x = int(width * 0.05)
        padding_y = int(height * 0.05)
        
        roi_img = img[padding_y:height-padding_y, padding_x:width-padding_x]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        puncte_albastre = []
        img_marcat = img.copy()
        
        cv2.rectangle(img_marcat, (padding_x, padding_y), 
                     (width-padding_x, height-padding_y), (255, 0, 255), 3)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 50 and area < 2000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx_roi = int(M["m10"] / M["m00"])
                    cy_roi = int(M["m01"] / M["m00"])
                    
                    cx = cx_roi + padding_x
                    cy = cy_roi + padding_y
                    
                    puncte_albastre.append((cx, cy))
                    
                    cv2.circle(img_marcat, (cx, cy), 15, (0, 0, 255), 3)
                    text = f"P{i+1}({cx},{cy})"
                    cv2.putText(img_marcat, text, (cx+20, cy-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if nume_output is None:
            nume_output = f"puncte_{datetime.now().strftime('%H%M%S')}.png"
        
        nume_mask = f"mask_{datetime.now().strftime('%H%M%S')}.png"
        
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[padding_y:height-padding_y, padding_x:width-padding_x] = mask
        
        cv2.imwrite(nume_output, img_marcat)
        cv2.imwrite(nume_mask, full_mask)
        
        return puncte_albastre
        
    except Exception as e:
        print(f"Eroare: {e}")
        return []

PUNCTE_CALIBRARE = {
    (749, 1563): [30, 85, 150, 115, 180, 90],
    (3445, 970): [30, 115, 130, 130, 180, 90],
    (502, 840): [30, 50, 140, 130, 170, 90],
    (2254, 310): [30, 45, 110, 160, 170, 90]
}

def distanta_euclidiana(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def este_punct_calibrare(coord, toleranta=120):
    for punct_calib in PUNCTE_CALIBRARE.keys():
        if distanta_euclidiana(coord, punct_calib) < toleranta:
            return True
    return False

def gaseste_pozitie_robot(coord_detectat):
    distante = []
    for punct_calib, pozitie_robot in PUNCTE_CALIBRARE.items():
        dist = distanta_euclidiana(coord_detectat, punct_calib)
        distante.append((dist, punct_calib, pozitie_robot))
    
    distante.sort(key=lambda x: x[0])
    
    if distante[0][0] < 30:
        return distante[0][2]
    
    puncte_apropiate = distante[:3]
    
    greutati = []
    suma_inv_dist = 0
    for dist, _, _ in puncte_apropiate:
        if dist < 1:
            dist = 1
        inv_dist = 1.0 / dist
        greutati.append(inv_dist)
        suma_inv_dist += inv_dist
    
    greutati = [g / suma_inv_dist for g in greutati]
    
    pozitie_noua = [0] * 6
    for i in range(6):
        for j, (_, _, pozitie) in enumerate(puncte_apropiate):
            pozitie_noua[i] += pozitie[i] * greutati[j]
        pozitie_noua[i] = int(round(pozitie_noua[i]))
    
    return pozitie_noua

def muta_robot_la_pozitie(pozitie_servouri, executa_actiune=True):
    try:
        comanda_deschis = f"{pozitie_servouri[0]},{pozitie_servouri[1]},{pozitie_servouri[2]},{pozitie_servouri[3]},{pozitie_servouri[4]},{pozitie_servouri[5]},10\n"
        comanda_inchis = f"{pozitie_servouri[0]},{pozitie_servouri[1]},{pozitie_servouri[2]},{pozitie_servouri[3]},{pozitie_servouri[4]},{pozitie_servouri[5]},70\n"
        
        if executa_actiune:
            ser.write(comanda_deschis.encode())
            time.sleep(3)
            ser.write(comanda_inchis.encode())
            time.sleep(10)
            ser.write(PozInitiala.encode())
            time.sleep(4)
        
        return True
        
    except Exception as e:
        print(f"Eroare robot: {e}")
        return False

def filtreaza_puncte_noi(puncte_detectate):
    puncte_noi = []
    puncte_calibrare_detectate = []
    
    for punct in puncte_detectate:
        if este_punct_calibrare(punct):
            puncte_calibrare_detectate.append(punct)
        else:
            puncte_noi.append(punct)
    
    return puncte_noi, puncte_calibrare_detectate

def analizeaza_si_muta_robot(simuleaza=False):
    nume_poza = face_poza("analiza_robot.png")
    
    if not os.path.exists(nume_poza):
        return None
    
    puncte_detectate = detecteaza_puncte_albastre(nume_poza, "robot_analiza.png")
    
    if not puncte_detectate:
        return None
    
    puncte_noi, puncte_calibrare = filtreaza_puncte_noi(puncte_detectate)
    
    if not puncte_noi:
        return None
    
    for i, punct in enumerate(puncte_noi, 1):
        pozitie_robot = gaseste_pozitie_robot(punct)
        success = muta_robot_la_pozitie(pozitie_robot, not simuleaza)
        
        if not simuleaza and i < len(puncte_noi):
            time.sleep(3)
    
    return puncte_noi

def test_sistem():
    nume_poza = face_poza("test_puncte.png")
    
    if os.path.exists(nume_poza):
        puncte = detecteaza_puncte_albastre(nume_poza, "rezultat_detectie.png")
        return puncte
    return []

def recalibreaza_punct(numar_punct, coord_noua, pozitie_noua):
    puncte_vechi = list(PUNCTE_CALIBRARE.keys())
    if 1 <= numar_punct <= 4:
        punct_vechi = puncte_vechi[numar_punct-1]
        del PUNCTE_CALIBRARE[punct_vechi]
        PUNCTE_CALIBRARE[coord_noua] = pozitie_noua
        print(f"Punct {numar_punct} recalibrat: {coord_noua} -> {pozitie_noua}")
        return True
    return False

def calibrare_rapida_pozitie():
    print("=== CALIBRARE RAPIDĂ POZIȚIE ===")
    afiseaza_calibrare_finala()
    
    try:
        punct_nr = int(input("Ce punct vrei să recalibrezi (1-4)? "))
        if punct_nr < 1 or punct_nr > 4:
            print("Punct invalid!")
            return
            
        puncte_keys = list(PUNCTE_CALIBRARE.keys())
        coord_curenta = puncte_keys[punct_nr-1]
        pozitie_curenta = PUNCTE_CALIBRARE[coord_curenta]
        
        print(f"Punctul {punct_nr}: {coord_curenta} -> {pozitie_curenta}")
        
        # Testează poziția curentă
        test = input("Testez poziția curentă? (y/n): ")
        if test.lower() == 'y':
            muta_robot_la_pozitie(pozitie_curenta, True)
        
        # Permite ajustarea fină
        print("Ajustare fină (lasă gol pentru a păstra valoarea):")
        pozitie_noua = pozitie_curenta.copy()
        
        for i, nume in enumerate(["Viteza", "m1(baza)", "m2(brat)", "m3(cot)", "m4(inchiet)", "m5(rotatie)"]):
            raspuns = input(f"{nume} [{pozitie_curenta[i]}]: ")
            if raspuns.strip():
                pozitie_noua[i] = int(raspuns)
        
        # Testează noua poziție
        print(f"Poziție nouă: {pozitie_noua}")
        test_nou = input("Testez noua poziție? (y/n): ")
        if test_nou.lower() == 'y':
            success = muta_robot_la_pozitie(pozitie_noua, True)
            if success:
                salvez = input("Salvez această poziție? (y/n): ")
                if salvez.lower() == 'y':
                    PUNCTE_CALIBRARE[coord_curenta] = pozitie_noua
                    print("Poziție salvată!")
                    
    except ValueError:
        print("Valoare invalidă!")

def ajusteaza_toate_pozitiile(offset_servouri):
    """Aplică un offset la toate pozitiule de calibrare"""
    for coord, pozitie in PUNCTE_CALIBRARE.items():
        for i in range(6):
            PUNCTE_CALIBRARE[coord][i] += offset_servouri[i]
        print(f"Ajustat {coord}: {pozitie}")

def calibrare_manuala():
    """Permite calibrarea manuală pas cu pas"""
    print("=== CALIBRARE MANUALĂ ===")
    
    for i, (coord, pozitie) in enumerate(PUNCTE_CALIBRARE.items(), 1):
        print(f"\nPunct {i}: {coord}")
        print(f"Poziție curentă: {pozitie}")
        
        while True:
            raspuns = input(f"Testez poziția pentru punctul {i}? (y/n/skip): ")
            if raspuns.lower() == 'y':
                muta_robot_la_pozitie(pozitie, True)
                
                corect = input("Poziția este corectă? (y/n): ")
                if corect.lower() == 'y':
                    break
                else:
                    print("Introdu noua poziție (6 valori separate prin virgulă):")
                    try:
                        input_pozitie = input("Viteza,m1,m2,m4,m3,m5: ")
                        pozitie_noua = [int(x.strip()) for x in input_pozitie.split(',')]
                        if len(pozitie_noua) == 6:
                            PUNCTE_CALIBRARE[coord] = pozitie_noua
                            print(f"Poziție actualizată: {pozitie_noua}")
                            break
                        else:
                            print("Trebuie să introduci exact 6 valori!")
                    except:
                        print("Format invalid! Folosește numere întregi separate prin virgulă.")
            elif raspuns.lower() == 'skip':
                break
            elif raspuns.lower() == 'n':
                return False
    
    print("\nCalibrare completată!")
    afiseaza_calibrare_finala()
    return True

def afiseaza_calibrare_finala():
    """Afișează calibrarea finală"""
    print("\n=== CALIBRARE FINALĂ ===")
    for i, (coord, pozitie) in enumerate(PUNCTE_CALIBRARE.items(), 1):
        print(f"Punct {i}: {coord} -> {pozitie}")

def ajustare_rapida():
    """Permite ajustarea rapidă a unui servo specific pentru toate punctele"""
    print("=== AJUSTARE RAPIDĂ ===")
    print("Servouri: 0=Viteza, 1=m1(baza), 2=m2(brat), 3=m3(cot), 4=m4(inchietura), 5=m5(rotatie), 6=m6(gripper)")
    
    try:
        servo_idx = int(input("Ce servo vrei să ajustezi (0-5)? "))
        if servo_idx < 0 or servo_idx > 5:
            print("Index servo invalid!")
            return
            
        offset = int(input(f"Cu cât să ajustez servo {servo_idx} (+/- grade)? "))
        
        print(f"Ajustez servo {servo_idx} cu {offset} grade pentru toate punctele...")
        
        for coord, pozitie in PUNCTE_CALIBRARE.items():
            pozitie[servo_idx] += offset
            print(f"Punct {coord}: servo {servo_idx} = {pozitie[servo_idx]}")
            
        print("Ajustare completată!")
        afiseaza_calibrare_finala()
        
    except ValueError:
        print("Valoare invalidă!")

def resetare_calibrare():
    """Resetează la valorile originale de calibrare"""
    global PUNCTE_CALIBRARE
    PUNCTE_CALIBRARE = {
        (749, 1563): [30, 85, 150, 115, 180, 90],
        (3445, 970): [30, 115, 130, 130, 180, 90],
        (502, 840): [30, 50, 140, 130, 170, 90],
        (2254, 310): [30, 45, 110, 160, 170, 90]
    }
    print("Calibrare resetată la valorile originale!")

def test_pozitie_manuala():
    print("=== TEST POZIȚIE MANUALĂ ===")
    print("Introduceți poziția servomotorului în format: Viteza,m1,m2,m4,m3,m5")
    
    try:
        input_pozitie = input("Poziție: ")
        pozitie = [int(x.strip()) for x in input_pozitie.split(',')]
        
        if len(pozitie) != 6:
            print("Trebuie să introduci exact 6 valori!")
            return
        
        print(f"Testez poziția: {pozitie}")
        confirmare = input("Continui? (y/n): ")
        
        if confirmare.lower() == 'y':
            success = muta_robot_la_pozitie(pozitie, True)
            if success:
                print("Poziție testată cu succes!")
                
                salvez = input("Salvez această poziție pentru un punct de calibrare? (y/n): ")
                if salvez.lower() == 'y':
                    afiseaza_calibrare_finala()
                    try:
                        numar_punct = int(input("Pentru ce punct (1-4)? "))
                        coord_x = int(input("Coordonata X: "))
                        coord_y = int(input("Coordonata Y: "))
                        
                        if 1 <= numar_punct <= 4:
                            puncte_keys = list(PUNCTE_CALIBRARE.keys())
                            punct_vechi = puncte_keys[numar_punct-1]
                            del PUNCTE_CALIBRARE[punct_vechi]
                            PUNCTE_CALIBRARE[(coord_x, coord_y)] = pozitie
                            print(f"Punctul {numar_punct} actualizat!")
                            afiseaza_calibrare_finala()
                    except:
                        print("Valori invalide!")
            else:
                print("Eroare la testarea poziției!")
                
    except ValueError:
        print("Format invalid! Folosește numere întregi separate prin virgulă.")

def main():
    print("SISTEM ROBOT - Alegeți modul:")
    print("1. Rulare normală (detectare și mișcare)")
    print("2. Calibrare manuală completă")
    print("3. Test detectare (fără robot)")
    print("4. Afișare puncte calibrare curente")
    print("5. Ajustare rapidă servouri")
    print("6. Resetare calibrare")
    print("7. Test poziție specifică")
    print("8. Calibrare rapidă punct specific")
    print("9. Salvare calibrare în fișier")
    print("10. Încărcare calibrare din fișier")
    
    try:
        alegere = input("Introduceți opțiunea (1-10): ").strip()
        
        if alegere == "1":
            puncte_procesate = analizeaza_si_muta_robot(simuleaza=False)
            if puncte_procesate:
                print(f"Procesate: {len(puncte_procesate)} puncte")
            else:
                print("Doar puncte de calibrare detectate")
                
        elif alegere == "2":
            calibrare_manuala()
            
        elif alegere == "3":
            puncte = test_sistem()
            print(f"Detectate {len(puncte)} puncte în modul test")
            
        elif alegere == "4":
            afiseaza_calibrare_finala()
            
        elif alegere == "5":
            ajustare_rapida()
            
        elif alegere == "6":
            resetare_calibrare()
            
        elif alegere == "7":
            test_pozitie_manuala()
            
        elif alegere == "8":
            calibrare_rapida_pozitie()
            
        elif alegere == "9":
            salveaza_calibrare()
            
        elif alegere == "10":
            incarca_calibrare()
            
        else:
            print("Opțiune invalidă")
            
    except KeyboardInterrupt:
        print("\nProgram întrerupt")

if __name__ == "__main__":
    main()


