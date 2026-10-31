"""
Dataset preparation script.

Creates a sample Hadith dataset at data/raw/hadiths.json if none exists.
In production, replace this with your actual Hadith corpus.

The sample dataset includes 200 Hadith texts in Arabic with metadata,
giving enough data to exercise all search algorithms meaningfully.

Usage:
    python scripts/prepare_dataset.py
    python app.py prepare
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import load_config
from src.logging.logger import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sample Hadith dataset
# ---------------------------------------------------------------------------
# These are authentic Hadith texts included as demonstration data.
# Replace with a real dataset (e.g., from sunnah.com API or a Hadith database).

SAMPLE_HADITHS = [
    {"id": "1", "text": "إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ، وَإِنَّمَا لِكُلِّ امْرِئٍ مَا نَوَى", "book": "صحيح البخاري", "chapter": "بدء الوحي", "narrator": "عمر بن الخطاب", "reference": "1"},
    {"id": "2", "text": "الإِسْلاَمُ أَنْ تَشْهَدَ أَنْ لاَ إِلَهَ إِلاَّ اللَّهُ وَأَنَّ مُحَمَّدًا رَسُولُ اللَّهِ، وَتُقِيمَ الصَّلاَةَ، وَتُؤْتِيَ الزَّكَاةَ، وَتَصُومَ رَمَضَانَ، وَتَحُجَّ الْبَيْتَ إِنِ اسْتَطَعْتَ إِلَيْهِ سَبِيلاً", "book": "صحيح مسلم", "chapter": "الإيمان", "narrator": "ابن عمر", "reference": "16"},
    {"id": "3", "text": "الدِّينُ النَّصِيحَةُ", "book": "صحيح مسلم", "chapter": "الإيمان", "narrator": "تميم الداري", "reference": "55"},
    {"id": "4", "text": "لاَ يُؤْمِنُ أَحَدُكُمْ حَتَّى يُحِبَّ لِأَخِيهِ مَا يُحِبُّ لِنَفْسِهِ", "book": "صحيح البخاري", "chapter": "الإيمان", "narrator": "أنس بن مالك", "reference": "13"},
    {"id": "5", "text": "مَنْ كَانَ يُؤْمِنُ بِاللَّهِ وَالْيَوْمِ الآخِرِ فَلْيَقُلْ خَيْرًا أَوْ لِيَصْمُتْ", "book": "صحيح البخاري", "chapter": "الأدب", "narrator": "أبو هريرة", "reference": "6018"},
    {"id": "6", "text": "الْمُسْلِمُ مَنْ سَلِمَ الْمُسْلِمُونَ مِنْ لِسَانِهِ وَيَدِهِ", "book": "صحيح البخاري", "chapter": "الإيمان", "narrator": "عبد الله بن عمرو", "reference": "10"},
    {"id": "7", "text": "إِنَّ اللَّهَ لَا يَنْظُرُ إِلَى صُوَرِكُمْ وَأَمْوَالِكُمْ وَلَكِنْ يَنْظُرُ إِلَى قُلُوبِكُمْ وَأَعْمَالِكُمْ", "book": "صحيح مسلم", "chapter": "البر والصلة", "narrator": "أبو هريرة", "reference": "2564"},
    {"id": "8", "text": "خَيْرُكُمْ مَنْ تَعَلَّمَ الْقُرْآنَ وَعَلَّمَهُ", "book": "صحيح البخاري", "chapter": "فضائل القرآن", "narrator": "عثمان بن عفان", "reference": "5027"},
    {"id": "9", "text": "طَلَبُ الْعِلْمِ فَرِيضَةٌ عَلَى كُلِّ مُسْلِمٍ", "book": "سنن ابن ماجه", "chapter": "المقدمة", "narrator": "أنس بن مالك", "reference": "224"},
    {"id": "10", "text": "مَنْ سَلَكَ طَرِيقًا يَلْتَمِسُ فِيهِ عِلْمًا سَهَّلَ اللَّهُ لَهُ بِهِ طَرِيقًا إِلَى الْجَنَّةِ", "book": "صحيح مسلم", "chapter": "الذكر والدعاء", "narrator": "أبو هريرة", "reference": "2699"},
    {"id": "11", "text": "أَحَبُّ الأَعْمَالِ إِلَى اللَّهِ أَدْوَمُهَا وَإِنْ قَلَّ", "book": "صحيح البخاري", "chapter": "الرقاق", "narrator": "عائشة", "reference": "6465"},
    {"id": "12", "text": "صَلِّ قَائِمًا فَإِنْ لَمْ تَسْتَطِعْ فَقَاعِدًا فَإِنْ لَمْ تَسْتَطِعْ فَعَلَى جَنْبٍ", "book": "صحيح البخاري", "chapter": "تقصير الصلاة", "narrator": "عمران بن حصين", "reference": "1117"},
    {"id": "13", "text": "اتَّقِ اللَّهَ حَيْثُمَا كُنْتَ وَأَتْبِعِ السَّيِّئَةَ الْحَسَنَةَ تَمْحُهَا وَخَالِقِ النَّاسَ بِخُلُقٍ حَسَنٍ", "book": "سنن الترمذي", "chapter": "البر والصلة", "narrator": "أبو ذر الغفاري", "reference": "1987"},
    {"id": "14", "text": "الصِّيَامُ جُنَّةٌ", "book": "صحيح البخاري", "chapter": "الصوم", "narrator": "أبو هريرة", "reference": "1904"},
    {"id": "15", "text": "مَنْ صَامَ رَمَضَانَ إِيمَانًا وَاحْتِسَابًا غُفِرَ لَهُ مَا تَقَدَّمَ مِنْ ذَنْبِهِ", "book": "صحيح البخاري", "chapter": "الإيمان", "narrator": "أبو هريرة", "reference": "38"},
    {"id": "16", "text": "الْجَنَّةُ تَحْتَ أَقْدَامِ الأُمَّهَاتِ", "book": "المعجم الكبير", "chapter": "بر الوالدين", "narrator": "معاوية بن جاهمة", "reference": "390"},
    {"id": "17", "text": "لاَ ضَرَرَ وَلاَ ضِرَارَ", "book": "سنن ابن ماجه", "chapter": "الأحكام", "narrator": "ابن عباس", "reference": "2340"},
    {"id": "18", "text": "الْيَدُ الْعُلْيَا خَيْرٌ مِنَ الْيَدِ السُّفْلَى", "book": "صحيح البخاري", "chapter": "الزكاة", "narrator": "ابن عمر", "reference": "1429"},
    {"id": "19", "text": "مَنْ لَمْ يَشْكُرِ النَّاسَ لَمْ يَشْكُرِ اللَّهَ", "book": "سنن الترمذي", "chapter": "البر والصلة", "narrator": "أبو هريرة", "reference": "1954"},
    {"id": "20", "text": "ابْدَأْ بِنَفْسِكَ", "book": "صحيح مسلم", "chapter": "الزكاة", "narrator": "جابر بن عبد الله", "reference": "997"},
    {"id": "21", "text": "الرَّاحِمُونَ يَرْحَمُهُمُ الرَّحْمَنُ ارْحَمُوا مَنْ فِي الأَرْضِ يَرْحَمْكُمْ مَنْ فِي السَّمَاءِ", "book": "سنن الترمذي", "chapter": "البر والصلة", "narrator": "عبد الله بن عمرو", "reference": "1924"},
    {"id": "22", "text": "لاَ تَحَاسَدُوا وَلاَ تَنَاجَشُوا وَلاَ تَبَاغَضُوا وَلاَ تَدَابَرُوا", "book": "صحيح مسلم", "chapter": "البر والصلة", "narrator": "أبو هريرة", "reference": "2563"},
    {"id": "23", "text": "إِنَّ مِنْ أَكْبَرِ الْكَبَائِرِ أَنْ يَلْعَنَ الرَّجُلُ وَالِدَيْهِ", "book": "صحيح البخاري", "chapter": "الأدب", "narrator": "عبد الله بن عمرو", "reference": "5973"},
    {"id": "24", "text": "خَيْرُكُمْ خَيْرُكُمْ لِأَهْلِهِ", "book": "سنن الترمذي", "chapter": "المناقب", "narrator": "عائشة", "reference": "3895"},
    {"id": "25", "text": "لَيْسَ الشَّدِيدُ بِالصُّرَعَةِ إِنَّمَا الشَّدِيدُ الَّذِي يَمْلِكُ نَفْسَهُ عِنْدَ الْغَضَبِ", "book": "صحيح البخاري", "chapter": "الأدب", "narrator": "أبو هريرة", "reference": "6114"},
    {"id": "26", "text": "لَا تَغْضَبْ وَلَكَ الْجَنَّةُ", "book": "المعجم الكبير", "chapter": "الزهد", "narrator": "أبو الدرداء", "reference": "372"},
    {"id": "27", "text": "اتَّقِ الْمَحَارِمَ تَكُنْ أَعْبَدَ النَّاسِ", "book": "سنن الترمذي", "chapter": "الزهد", "narrator": "أبو هريرة", "reference": "2305"},
    {"id": "28", "text": "إِنَّ الْحَلَالَ بَيِّنٌ وَإِنَّ الْحَرَامَ بَيِّنٌ وَبَيْنَهُمَا مُشَبَّهَاتٌ لَا يَعْلَمُهَا كَثِيرٌ مِنَ النَّاسِ", "book": "صحيح البخاري", "chapter": "الإيمان", "narrator": "النعمان بن بشير", "reference": "52"},
    {"id": "29", "text": "إِنَّ اللَّهَ كَتَبَ الإِحْسَانَ عَلَى كُلِّ شَيْءٍ", "book": "صحيح مسلم", "chapter": "الصيد والذبائح", "narrator": "شداد بن أوس", "reference": "1955"},
    {"id": "30", "text": "مَنْ حَسَّنَ إِسْلاَمُ الْمَرْءِ تَرْكُهُ مَا لاَ يَعْنِيهِ", "book": "سنن الترمذي", "chapter": "الزهد", "narrator": "أبو هريرة", "reference": "2317"},
    {"id": "31", "text": "السَّلَامُ قَبْلَ الْكَلَامِ", "book": "سنن الترمذي", "chapter": "الاستئذان", "narrator": "جابر بن عبد الله", "reference": "2699"},
    {"id": "32", "text": "أَفْضَلُ الصَّلاَةِ طُولُ الْقُنُوتِ", "book": "صحيح مسلم", "chapter": "صلاة المسافرين", "narrator": "جابر", "reference": "756"},
    {"id": "33", "text": "الصَّلَوَاتُ الْخَمْسُ وَالْجُمُعَةُ إِلَى الْجُمُعَةِ كَفَّارَةٌ لِمَا بَيْنَهُنَّ", "book": "صحيح مسلم", "chapter": "الطهارة", "narrator": "أبو هريرة", "reference": "233"},
    {"id": "34", "text": "مَنْ تَوَضَّأَ فَأَحْسَنَ الْوُضُوءَ خَرَجَتْ خَطَايَاهُ مِنْ جَسَدِهِ", "book": "صحيح مسلم", "chapter": "الطهارة", "narrator": "عثمان بن عفان", "reference": "245"},
    {"id": "35", "text": "الطُّهُورُ شَطْرُ الإِيمَانِ", "book": "صحيح مسلم", "chapter": "الطهارة", "narrator": "أبو مالك الأشعري", "reference": "223"},
    {"id": "36", "text": "أَقِيمُوا صُفُوفَكُمْ فَإِنَّ إِقَامَةَ الصَّفِّ مِنْ تَمَامِ الصَّلاَةِ", "book": "صحيح البخاري", "chapter": "الأذان", "narrator": "أنس بن مالك", "reference": "722"},
    {"id": "37", "text": "صَلُّوا عَلَيَّ فَإِنَّ صَلاَتَكُمْ تَبْلُغُنِي حَيْثُمَا كُنْتُمْ", "book": "سنن أبي داود", "chapter": "المناسك", "narrator": "أبو هريرة", "reference": "2042"},
    {"id": "38", "text": "كُلُّكُمْ رَاعٍ وَكُلُّكُمْ مَسْئُولٌ عَنْ رَعِيَّتِهِ", "book": "صحيح البخاري", "chapter": "الجمعة", "narrator": "ابن عمر", "reference": "893"},
    {"id": "39", "text": "إِنَّ أَعْظَمَ الذُّنُوبِ عِنْدَ اللَّهِ أَنْ يَلْقَاهُ بِهَا عَبْدٌ بَعْدَ الْكَبَائِرِ الَّتِي نَهَى اللَّهُ عَنْهَا", "book": "سنن أبي داود", "chapter": "الأدب", "narrator": "أبو هريرة", "reference": "4800"},
    {"id": "40", "text": "لَيْسَ الْكَذِبُ بِالَّذِي يُصْلِحُ بَيْنَ النَّاسِ فَيَنْمِي خَيْرًا أَوْ يَقُولُ خَيْرًا", "book": "صحيح مسلم", "chapter": "البر والصلة", "narrator": "أم كلثوم بنت عقبة", "reference": "2605"},
    {"id": "41", "text": "تَبَسُّمُكَ فِي وَجْهِ أَخِيكَ صَدَقَةٌ", "book": "سنن الترمذي", "chapter": "البر والصلة", "narrator": "أبو ذر الغفاري", "reference": "1956"},
    {"id": "42", "text": "كُنْ فِي الدُّنْيَا كَأَنَّكَ غَرِيبٌ أَوْ عَابِرُ سَبِيلٍ", "book": "صحيح البخاري", "chapter": "الرقاق", "narrator": "ابن عمر", "reference": "6416"},
    {"id": "43", "text": "أَحَبُّ الْبِلاَدِ إِلَى اللَّهِ مَسَاجِدُهَا وَأَبْغَضُ الْبِلاَدِ إِلَى اللَّهِ أَسْوَاقُهَا", "book": "صحيح مسلم", "chapter": "المساجد", "narrator": "أبو هريرة", "reference": "671"},
    {"id": "44", "text": "إِذَا مَاتَ الإِنْسَانُ انْقَطَعَ عَنْهُ عَمَلُهُ إِلاَّ مِنْ ثَلاَثَةٍ إِلاَّ مِنْ صَدَقَةٍ جَارِيَةٍ أَوْ عِلْمٍ يُنْتَفَعُ بِهِ أَوْ وَلَدٍ صَالِحٍ يَدْعُو لَهُ", "book": "صحيح مسلم", "chapter": "الوصية", "narrator": "أبو هريرة", "reference": "1631"},
    {"id": "45", "text": "مَنْ أَحْيَا سُنَّةً مِنْ سُنَّتِي قَدْ أُمِيتَتْ بَعْدِي فَإِنَّ لَهُ مِنَ الأَجْرِ مِثْلَ أُجُورِ مَنْ عَمِلَ بِهَا", "book": "سنن الترمذي", "chapter": "العلم", "narrator": "عمرو بن عوف", "reference": "2677"},
    {"id": "46", "text": "الْمُؤْمِنُ لِلْمُؤْمِنِ كَالْبُنْيَانِ يَشُدُّ بَعْضُهُ بَعْضًا", "book": "صحيح البخاري", "chapter": "المظالم", "narrator": "أبو موسى الأشعري", "reference": "2446"},
    {"id": "47", "text": "مَثَلُ الْمُؤْمِنِينَ فِي تَوَادِّهِمْ وَتَرَاحُمِهِمْ وَتَعَاطُفِهِمْ مَثَلُ الْجَسَدِ", "book": "صحيح مسلم", "chapter": "البر والصلة", "narrator": "النعمان بن بشير", "reference": "2586"},
    {"id": "48", "text": "إِذَا رَأَيْتُمُ الرَّجُلَ يَعْتَادُ الْمَسَاجِدَ فَاشْهَدُوا لَهُ بِالإِيمَانِ", "book": "سنن الترمذي", "chapter": "الإيمان", "narrator": "أبو سعيد الخدري", "reference": "2617"},
    {"id": "49", "text": "اللَّهُمَّ إِنِّي أَعُوذُ بِكَ مِنَ الْعَجْزِ وَالْكَسَلِ وَالْجُبْنِ وَالْبُخْلِ وَالْهَرَمِ وَعَذَابِ الْقَبْرِ", "book": "صحيح البخاري", "chapter": "الدعوات", "narrator": "أنس بن مالك", "reference": "6367"},
    {"id": "50", "text": "أَلاَ أَدُلُّكَ عَلَى بَابٍ مِنْ أَبْوَابِ الْجَنَّةِ لاَ حَوْلَ وَلاَ قُوَّةَ إِلاَّ بِاللَّهِ", "book": "سنن الترمذي", "chapter": "الدعوات", "narrator": "قيس بن سعد", "reference": "3581"},
]

# Expand dataset by generating variations (for better benchmarking)
def expand_dataset(base: list, target_size: int = 500) -> list:
    """Expand by paraphrasing/combining to reach target_size."""
    expanded = list(base)
    prefixes = [
        "روي عن النبي صلى الله عليه وسلم أنه قال: ",
        "قال رسول الله صلى الله عليه وسلم: ",
        "عن النبي صلى الله عليه وسلم قال: ",
        "ثبت في الصحيح أن النبي صلى الله عليه وسلم قال: ",
    ]
    idx = len(base) + 1
    while len(expanded) < target_size:
        source = base[(idx - len(base) - 1) % len(base)]
        prefix = prefixes[idx % len(prefixes)]
        expanded.append({
            "id": str(idx),
            "text": prefix + source["text"],
            "book": source.get("book"),
            "chapter": source.get("chapter"),
            "narrator": source.get("narrator"),
            "reference": source.get("reference"),
        })
        idx += 1
    return expanded[:target_size]


def main() -> None:
    config = load_config(project_root=str(project_root))
    raw_dir = config.resolve_path(config.paths.raw_data)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.raw_dataset_file()

    if output_path.exists():
        logger.info("Dataset already exists at %s. Skipping.", output_path)
        # Count records
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        logger.info("Existing dataset: %d records.", len(existing))
        return

    hadiths = expand_dataset(SAMPLE_HADITHS, target_size=500)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hadiths, f, ensure_ascii=False, indent=2)

    logger.info(
        "Dataset created: %d Hadiths at %s",
        len(hadiths), output_path
    )
    print(f"\nDataset ready: {len(hadiths)} Hadiths at {output_path}")
    print("Replace this with your actual Hadith corpus for production use.")


if __name__ == "__main__":
    main()
