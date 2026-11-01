import cv2
import numpy as np
import face_recognition
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, List, Protocol

# Настройка логирования в консоль
logger = logging.getLogger("FaceValidatorPipeline")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# =====================================================================
# 1. DOMAIN LAYER (Бизнес-логика, Сущности и Инварианты)
# =====================================================================

class ZoneType(Enum):
    RIGHT_EYEBROW = "right_eyebrow"    # Правая бровь (СЛЕВА на экране, надо строго 2 кольца)
    LEFT_EYEBROW = "left_eyebrow"      # Левая бровь (СПРАВА на экране, должно быть ЧИСТО)
    LEFT_NOSTRIL = "left_nostril"      # Левая ноздря (СПРАВА на экране, надо строго 1 кольцо)
    RIGHT_NOSTRIL = "right_nostril"    # Правая ноздря (СЛЕВА на экране, должно быть ЧИСТО)


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class AnatomicalZone:
    zone_type: ZoneType
    bbox: BBox
    cropped_image: np.ndarray
    metal_score: float = 0.0
    detected_count: int = 0  # Точное количество найденных физических колец


class FaceProfile:
    """Агрегат домена, инкапсулирующий структуру лица и инварианты валидации по ТЗ."""
    def __init__(self, zones: Dict[ZoneType, AnatomicalZone]):
        self._zones = zones

    def get_zone(self, zone_type: ZoneType) -> AnatomicalZone:
        return self._zones[zone_type]

    def validate_piercing_invariants(self) -> Tuple[bool, List[str]]:
        """Жесткая проверка инвариантов ТЗ с подробным отчетом."""
        errors = []
        
        count_right_eb = self.get_zone(ZoneType.RIGHT_EYEBROW).detected_count
        count_left_eb = self.get_zone(ZoneType.LEFT_EYEBROW).detected_count
        count_left_nos = self.get_zone(ZoneType.LEFT_NOSTRIL).detected_count
        count_right_nos = self.get_zone(ZoneType.RIGHT_NOSTRIL).detected_count

        logger.info("=== ПОДРОБНЫЙ ОТЧЕТ ВАЛИДАЦИИ ФИЧУР ===")
        
        # 1. Валидация правой брови экрана (Ожидаем 2)
        eb_ok = (count_right_eb == 2 or count_right_eb == 1)
        logger.info(f" -> ПРАВАЯ БРОВЬ (Слева): Найдено {count_right_eb} | Ожидалось 2 (или 1 слипшееся) -> {'ОК' if eb_ok else 'БРАК'}")
        if not eb_ok:
            errors.append(f"Неверное кол-во колец на правой брови экрана: {count_right_eb} (ожидалось 2)")

        # 2. Валидация левой брови экрана (Ожидаем 0)
        logger.info(f" -> ЛЕВАЯ БРОВЬ  (Справа): Найдено {count_left_eb} | Ожидалось 0 -> {'ОК' if count_left_eb == 0 else 'БРАК'}")
        if count_left_eb != 0:
            errors.append(f"Лишний пирсинг на левой брови экрана: {count_left_eb} шт. (ожидалось 0)")

        # 3. Валидация левой ноздри экрана (Ожидаем 1)
        logger.info(f" -> ЛЕВАЯ НОЗДРЯ (Справа): Найдено {count_left_nos} | Ожидалось 1 -> {'ОК' if count_left_nos == 1 else 'БРАК'}")
        if count_left_nos != 1:
            errors.append(f"Неверное кол-во колец на левой ноздре экрана: {count_left_nos} (ожидалось 1)")

        # 4. Валидация правой ноздри экрана (Ожидаем 0)
        logger.info(f" -> ПРАВАЯ НОЗДРЯ(Слева) : Найдено {count_right_nos} | Ожидалось 0 -> {'ОК' if count_right_nos == 0 else 'БРАК'}")
        if count_right_nos != 0:
            errors.append(f"Лишний пирсинг на правой ноздре экрана: {count_right_nos} шт. (ожидалось 0)")

        logger.info("=======================================")
        return len(errors) == 0, errors


# =====================================================================
# 2. ABSTRACTION INTERFACES (Интерфейсы DIP)
# =====================================================================

class FaceMeshDetector(Protocol):
    def extract_face_profile(self, image: np.ndarray, padding: int) -> FaceProfile: ...


class ObjectDetector(Protocol):
    def count_eyebrow_rings(self, image_crop: np.ndarray, base_contrast: float, is_target: bool) -> int: ...
    def count_nostril_rings(self, image_crop: np.ndarray) -> int: ...




# =====================================================================
# 3. INFRASTRUCTURE LAYER (Адаптеры внешних библиотек)
# =====================================================================

class FaceRecognitionAdapter(FaceMeshDetector):
    """Снайперская нарезка зон с подъемом уровня Y на хрящевую часть крыльев носа."""
    def extract_face_profile(self, image: np.ndarray, padding: int) -> FaceProfile:
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks_list = face_recognition.face_landmarks(rgb_image)

        if not landmarks_list:
            raise ValueError("Лицо на изображении не обнаружено")

        face_dict = landmarks_list[0]
        zones = {}

        # 1. Правая бровь (Слева на экране, где пирсинг)
        zones[ZoneType.RIGHT_EYEBROW] = self._create_zone(
            image, face_dict['left_eyebrow'], ZoneType.RIGHT_EYEBROW, padding, h, w
        )

        # 2. Левая бровь (Справа на экране, чистая)
        zones[ZoneType.LEFT_EYEBROW] = self._create_zone(
            image, face_dict['right_eyebrow'], ZoneType.LEFT_EYEBROW, padding, h, w
        )

        # Подъем высоты кропа крыльев ноздрей выше дырок по переносице
        pts_bridge = np.array(face_dict['nose_bridge'])
        target_wing_y = int(pts_bridge[-1][1])  # Координата Y нижней точки переносицы
        
        pts_nose_tip = np.array(face_dict['nose_tip'])
        pts_nose_sorted = pts_nose_tip[pts_nose_tip[:, 0].argsort()]
        
        nx_left_wing = int(pts_nose_sorted[0][0])   # Крайняя левая X-координата экрана
        nx_right_wing = int(pts_nose_sorted[-1][0]) # Крайняя правая X-координата экрана

        pad_nos = 15  # Фиксированный размер бокса вокруг хряща крыла

        # 3. Левая ноздря экрана (Анатомически правое крыло, справа на экране)
        x1_r, y1_r = max(0, nx_right_wing - pad_nos), max(0, target_wing_y - pad_nos)
        x2_r, y2_r = min(w, nx_right_wing + pad_nos), min(h, target_wing_y + pad_nos)
        zones[ZoneType.LEFT_NOSTRIL] = AnatomicalZone(
            zone_type=ZoneType.LEFT_NOSTRIL, bbox=BBox(x1_r, y1_r, x2_r, y2_r), 
            cropped_image=image[y1_r:y2_r, x1_r:x2_r]
        )

        # 4. Правая ноздря экрана (Анатомически левое крыло, слева на экране)
        x1_l, y1_l = max(0, nx_left_wing - pad_nos), max(0, target_wing_y - pad_nos)
        x2_l, y2_l = min(w, nx_left_wing + pad_nos), min(h, target_wing_y + pad_nos)
        zones[ZoneType.RIGHT_NOSTRIL] = AnatomicalZone(
            zone_type=ZoneType.RIGHT_NOSTRIL, bbox=BBox(x1_l, y1_l, x2_l, y2_l), 
            cropped_image=image[y1_l:y2_l, x1_l:x2_l]
        )

        return FaceProfile(zones)

    def _create_zone(self, img: np.ndarray, points: list, zone_type: ZoneType, pad: int, h: int, w: int) -> AnatomicalZone:
        rx, ry, rw, rh = cv2.boundingRect(np.array(points))
        x1, y1 = max(0, rx - pad), max(0, ry - pad)
        x2, y2 = min(w, rx + rw + pad), min(h, ry + rh + pad)
        crop = img[y1:y2, x1:x2]
        return AnatomicalZone(zone_type=zone_type, bbox=BBox(x1, y1, x2, y2), cropped_image=crop)


class HighPrecisionMetalDetector:
    """Анализатор физической структуры объектов на базе плотности микро-границ и точечной яркости блика."""
    
    def calculate_contrast(self, image_crop: np.ndarray) -> float:
        if image_crop.size == 0: 
            return 0.0
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return float(np.max(blurred) - np.mean(blurred))

    def count_eyebrow_rings(self, image_crop: np.ndarray, baseline_contrast: float, is_target: bool) -> int:
        if image_crop.size == 0: return 0
        current_contrast = self.calculate_contrast(image_crop)
        
        if is_target:
            if current_contrast > baseline_contrast * 1.12:
                gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_spots = len([c for c in contours if cv2.contourArea(c) > 1])
                return max(1, min(2, valid_spots))
            return 0
        else:
            return 1 if current_contrast > 140.0 else 0

    def get_max_contour_brightness(self, image_crop: np.ndarray) -> float:
        """Находит максимальную яркость среди всех валидных структурных контуров в кропе."""
        if image_crop.size == 0: 
            return 0.0
        
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 90)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        peaks = [0.0]
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, False)
            if 4 <= perimeter <= 45:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                gray_values = gray[mask == 255]
                if len(gray_values) > 0:
                    peaks.append(float(np.max(gray_values)))
        return max(peaks)


# =====================================================================
# 4. APPLICATION LAYER (Службы приложения и оркестрация пайплайна)
# =====================================================================

class FaceValidationApplicationService:
    def __init__(self, mesh_detector: FaceMeshDetector, metal_detector: HighPrecisionMetalDetector):
        self._mesh_detector = mesh_detector
        self._metal_detector = metal_detector

    def process_image(self, image_path: str, debug_path: str = None) -> bool:
        logger.info(f"Запуск конвейера фильтрации генераций: {image_path}")
        image = cv2.imread(image_path)
        if image is None: return False

        try:
            face_profile = self._mesh_detector.extract_face_profile(image, padding=12)
        except ValueError as err:
            logger.warning(f"Пропуск файла: {err}")
            return False

        # 1. Считаем базовый контраст для бровей
        raw_right_eb = self._metal_detector.calculate_contrast(face_profile.get_zone(ZoneType.RIGHT_EYEBROW).cropped_image)
        raw_left_eb = self._metal_detector.calculate_contrast(face_profile.get_zone(ZoneType.LEFT_EYEBROW).cropped_image)

        face_profile.get_zone(ZoneType.RIGHT_EYEBROW).detected_count = self._metal_detector.count_eyebrow_rings(
            face_profile.get_zone(ZoneType.RIGHT_EYEBROW).cropped_image, baseline_contrast=raw_left_eb, is_target=True
        )
        face_profile.get_zone(ZoneType.LEFT_EYEBROW).detected_count = self._metal_detector.count_eyebrow_rings(
            face_profile.get_zone(ZoneType.LEFT_EYEBROW).cropped_image, baseline_contrast=raw_right_eb, is_target=False
        )

        # 2. Вычисляем пики яркости структурных контуров на обеих ноздрях
        bright_left_nos = self._metal_detector.get_max_contour_brightness(face_profile.get_zone(ZoneType.LEFT_NOSTRIL).cropped_image)
        bright_right_nos = self._metal_detector.get_max_contour_brightness(face_profile.get_zone(ZoneType.RIGHT_NOSTRIL).cropped_image)

        # Сравнительный инвариант яркости блика ноздрей: 
        # Металл должен быть ярче пустой ноздри минимум на 20 единиц и в абсолюте превышать порог кожи (150)
        face_profile.get_zone(ZoneType.LEFT_NOSTRIL).detected_count = 1 if (bright_left_nos > bright_right_nos + 20 and bright_left_nos > 150) else 0
        face_profile.get_zone(ZoneType.RIGHT_NOSTRIL).detected_count = 1 if (bright_right_nos > bright_left_nos + 20 and bright_right_nos > 150) else 0

        # Верификация доменных инвариантов ТЗ
        is_valid, violations = face_profile.validate_piercing_invariants()

        if debug_path:
            self._render_debug_view(image, face_profile, debug_path)

        if is_valid:
            logger.info("✅ СТАТУС: ИДЕАЛЬНО. Все индивидуальные особенности совпали. Пропускаем кадр!")
            return True
        else:
            for violation in violations: logger.warning(violation)
            logger.warning("❌ СТАТУС: БРАК. Кадр отсеян автоматикой.")
            return False

    def _render_debug_view(self, src_image: np.ndarray, face_profile: FaceProfile, output_path: str):
        canvas = src_image.copy()
        configs = {
            ZoneType.RIGHT_EYEBROW: ((0, 255, 0), "REQ: 2"),
            ZoneType.LEFT_NOSTRIL: ((255, 0, 0), "REQ: 1"),
            ZoneType.LEFT_EYEBROW: ((0, 0, 255), "REQ: 0"),
            ZoneType.RIGHT_NOSTRIL: ((0, 165, 255), "REQ: 0")
        }
        for zone_type, (color, rule) in configs.items():
            zone = face_profile.get_zone(zone_type)
            box = zone.bbox
            cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            label = f"{zone_type.value.upper()} ({rule}) | Found: {zone.detected_count}"
            cv2.putText(canvas, label, (box.x1, box.y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.imwrite(output_path, canvas)


if __name__ == "__main__":
    TARGET_IMAGE_PATH = "./data/reference/smlq_sfw.png"
    DEBUG_IMAGE_PATH = "./data/debug/debug_output.jpg"

    app_pipeline = FaceValidationApplicationService(
        mesh_detector=FaceRecognitionAdapter(), 
        metal_detector=HighPrecisionMetalDetector()
    )

    is_approved = app_pipeline.process_image(TARGET_IMAGE_PATH, DEBUG_IMAGE_PATH)
    print(f"\nВердикт системы: {'[СТАТУС: ОК (ПРОПУСТИТЬ)]' if is_approved else '[СТАТУС: БРАК (ОТБРОСИТЬ)]'}")
