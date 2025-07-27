import logging
import re
from typing import Dict, List, Any
from database.models import get_db_session, UserFeedback, FieldExtraction, Invoice
from datetime import datetime

logger = logging.getLogger(__name__)


class LearningSystem:
    """Enhanced learning system that improves extractions in real-time"""

    def __init__(self):
        self.improvement_threshold = 0.8
        self.learning_rate = 0.1
        self.learned_patterns = {}
        self._load_learned_patterns()

    def _load_learned_patterns(self):
        """Load previously learned patterns from user corrections"""
        try:
            db_session = get_db_session()

            # Get all corrections to learn from
            corrections = db_session.query(UserFeedback).filter(
                UserFeedback.feedback_type == 'correction'
            ).all()

            # Group corrections by field
            field_corrections = {}
            for correction in corrections:
                field_name = correction.field_name
                if field_name not in field_corrections:
                    field_corrections[field_name] = []

                field_corrections[field_name].append({
                    'original': correction.original_value,
                    'corrected': correction.corrected_value,
                    'confidence': correction.confidence_before
                })

            # Learn patterns from corrections
            for field_name, corrections_list in field_corrections.items():
                self.learned_patterns[field_name] = self._extract_patterns_from_corrections(
                    corrections_list
                )

            logger.info(f"Loaded learned patterns for {len(self.learned_patterns)} fields")

        except Exception as e:
            logger.error(f"Error loading learned patterns: {e}")
        finally:
            if 'db_session' in locals():
                db_session.close()

    def _extract_patterns_from_corrections(self, corrections_list):
        """Extract patterns from user corrections"""
        patterns = {
            'common_corrections': {},
            'improved_patterns': [],
            'confidence_boosters': []
        }

        for correction in corrections_list:
            original = correction['original'].strip()
            corrected = correction['corrected'].strip()

            if original != corrected:
                # Store common correction mapping
                patterns['common_corrections'][original.lower()] = corrected

                # If original was empty but user provided value, learn the pattern
                if not original and corrected:
                    patterns['confidence_boosters'].append(corrected)

                # Learn pattern improvements
                if len(corrected) > 2:
                    patterns['improved_patterns'].append(corrected)

        return patterns

    def apply_learned_patterns(self, field_extractor, filename):
        """Apply learned patterns to improve extraction before processing"""

        # Enhance field extractor with learned patterns
        enhanced_extractor = EnhancedFieldExtractor(field_extractor, self.learned_patterns)

        logger.info(f"Applied learned patterns for {filename}")
        return enhanced_extractor

    def save_field_corrections(self, invoice_id: int, original_fields: Dict, corrected_fields: Dict) -> bool:
        """Save user corrections and immediately update learning patterns"""

        try:
            db_session = get_db_session()

            corrections_made = 0
            new_learnings = []

            # Compare each field and save corrections
            for field_name in original_fields.keys():
                original_value = str(original_fields[field_name]['value']).strip()
                corrected_value = str(corrected_fields.get(field_name, original_value)).strip()

                # Check if user made a correction
                if original_value != corrected_value:
                    corrections_made += 1

                    # Save individual field correction
                    feedback = UserFeedback(
                        invoice_id=invoice_id,
                        field_name=field_name,
                        original_value=original_value,
                        corrected_value=corrected_value,
                        feedback_type='correction',
                        confidence_before=original_fields[field_name]['confidence'],
                        user_rating=None,
                        is_used_for_training=True  # Immediately available for learning
                    )

                    db_session.add(feedback)

                    # Record the learning for immediate application
                    new_learnings.append({
                        'field': field_name,
                        'original': original_value,
                        'corrected': corrected_value,
                        'confidence': original_fields[field_name]['confidence']
                    })

            # Save complete field extraction record
            field_extraction = FieldExtraction(
                invoice_id=invoice_id,

                # Original extractions
                invoice_number_extracted=str(original_fields.get('invoice_number', {}).get('value', '')),
                invoice_date_extracted=str(original_fields.get('date', {}).get('value', '')),
                supplier_name_extracted=str(original_fields.get('supplier', {}).get('value', '')),
                total_amount_extracted=float(original_fields.get('total', {}).get('value', 0)),
                vat_amount_extracted=float(original_fields.get('vat', {}).get('value', 0)),

                # User corrections
                invoice_number_corrected=str(
                    corrected_fields.get('invoice_number', original_fields.get('invoice_number', {}).get('value', ''))),
                invoice_date_corrected=str(
                    corrected_fields.get('date', original_fields.get('date', {}).get('value', ''))),
                supplier_name_corrected=str(
                    corrected_fields.get('supplier', original_fields.get('supplier', {}).get('value', ''))),
                total_amount_corrected=float(
                    corrected_fields.get('total', original_fields.get('total', {}).get('value', 0))),
                vat_amount_corrected=float(corrected_fields.get('vat', original_fields.get('vat', {}).get('value', 0))),

                # Metadata
                feedback_provided=True,
                correction_count=corrections_made,
                feedback_date=datetime.utcnow()
            )

            db_session.add(field_extraction)
            db_session.commit()

            # Immediately update learned patterns with new corrections
            self._update_learned_patterns(new_learnings)

            logger.info(f"Saved {corrections_made} corrections for invoice {invoice_id} and updated learning patterns")
            return True

        except Exception as e:
            logger.error(f"Error saving corrections: {e}")
            if 'db_session' in locals():
                db_session.rollback()
            return False
        finally:
            if 'db_session' in locals():
                db_session.close()

    def _update_learned_patterns(self, new_learnings):
        """Update learned patterns with new corrections immediately"""

        for learning in new_learnings:
            field_name = learning['field']
            original = learning['original']
            corrected = learning['corrected']

            # Initialize field patterns if not exists
            if field_name not in self.learned_patterns:
                self.learned_patterns[field_name] = {
                    'common_corrections': {},
                    'improved_patterns': [],
                    'confidence_boosters': []
                }

            patterns = self.learned_patterns[field_name]

            # Add to common corrections
            if original:
                patterns['common_corrections'][original.lower()] = corrected

            # Add to confidence boosters if it was a new detection
            if not original and corrected:
                if corrected not in patterns['confidence_boosters']:
                    patterns['confidence_boosters'].append(corrected)

            # Add to improved patterns
            if len(corrected) > 2 and corrected not in patterns['improved_patterns']:
                patterns['improved_patterns'].append(corrected)

        logger.info(f"Updated learned patterns with {len(new_learnings)} new corrections")

    def get_field_statistics(self) -> Dict:
        """Get statistics about field extraction performance"""
        try:
            db_session = get_db_session()

            # Get all field extractions
            extractions = db_session.query(FieldExtraction).all()

            if not extractions:
                return {
                    'total_extractions': 0,
                    'total_corrections': 0,
                    'accuracy_rate': 0.0,
                    'most_problematic_fields': [],
                    'improvement_trend': 'No data'
                }

            total_extractions = len(extractions)
            total_corrections = sum(e.correction_count for e in extractions)

            # Calculate accuracy rate
            accuracy_rate = 1.0 - (total_corrections / (total_extractions * 8))  # 8 main fields
            accuracy_rate = max(0.0, accuracy_rate)  # Ensure non-negative

            # Find most problematic fields
            field_errors = {}
            for extraction in extractions:
                if extraction.invoice_number_extracted != extraction.invoice_number_corrected:
                    field_errors['invoice_number'] = field_errors.get('invoice_number', 0) + 1
                if extraction.invoice_date_extracted != extraction.invoice_date_corrected:
                    field_errors['date'] = field_errors.get('date', 0) + 1
                if extraction.supplier_name_extracted != extraction.supplier_name_corrected:
                    field_errors['supplier'] = field_errors.get('supplier', 0) + 1
                if abs(extraction.total_amount_extracted - extraction.total_amount_corrected) > 0.01:
                    field_errors['total_amount'] = field_errors.get('total_amount', 0) + 1

            # Sort by error count
            problematic_fields = sorted(field_errors.items(), key=lambda x: x[1], reverse=True)

            return {
                'total_extractions': total_extractions,
                'total_corrections': total_corrections,
                'accuracy_rate': accuracy_rate,
                'most_problematic_fields': problematic_fields[:3],
                'improvement_trend': 'Improving' if accuracy_rate > 0.7 else 'Needs attention'
            }

        except Exception as e:
            logger.error(f"Error getting field statistics: {e}")
            return {
                'total_extractions': 0,
                'total_corrections': 0,
                'accuracy_rate': 0.0,
                'most_problematic_fields': [],
                'improvement_trend': 'Error loading data'
            }
        finally:
            if 'db_session' in locals():
                db_session.close()

    def get_learning_patterns(self) -> Dict:
        """Analyze correction patterns to improve extraction"""
        try:
            db_session = get_db_session()

            # Get all corrections
            corrections = db_session.query(UserFeedback).filter(
                UserFeedback.feedback_type == 'correction'
            ).all()

            # Analyze patterns
            patterns = {
                'total_corrections': len(corrections),
                'field_accuracy': {},
                'common_mistakes': {},
                'improvement_suggestions': []
            }

            # Calculate accuracy per field
            field_counts = {}
            field_correct = {}

            for correction in corrections:
                field_name = correction.field_name

                field_counts[field_name] = field_counts.get(field_name, 0) + 1

                # If original was wrong (needed correction), mark as incorrect
                if correction.original_value != correction.corrected_value:
                    field_correct[field_name] = field_correct.get(field_name, 0)
                else:
                    field_correct[field_name] = field_correct.get(field_name, 0) + 1

            # Calculate accuracy rates
            for field_name in field_counts:
                if field_counts[field_name] > 0:
                    accuracy = field_correct.get(field_name, 0) / field_counts[field_name]
                    patterns['field_accuracy'][field_name] = accuracy

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing learning patterns: {e}")
            return {'error': str(e)}
        finally:
            if 'db_session' in locals():
                db_session.close()


class EnhancedFieldExtractor:
    """Enhanced field extractor that uses learned patterns"""

    def __init__(self, base_extractor, learned_patterns):
        self.base_extractor = base_extractor
        self.learned_patterns = learned_patterns

    def extract_all_fields(self, text: str) -> Dict:
        """Extract fields using base extractor + learned patterns"""

        # Get base extraction
        base_results = self.base_extractor.extract_all_fields(text)

        # Enhance with learned patterns
        enhanced_results = {}

        for field_name, field_data in base_results.items():
            enhanced_results[field_name] = self._enhance_field_extraction(
                field_name, field_data, text
            )

        return enhanced_results

    def _enhance_field_extraction(self, field_name, field_data, text):
        """Enhance single field extraction with learned patterns"""

        if field_name not in self.learned_patterns:
            return field_data

        patterns = self.learned_patterns[field_name]
        original_value = str(field_data['value']).strip()
        original_confidence = field_data['confidence']

        # Apply common corrections
        if original_value.lower() in patterns['common_corrections']:
            corrected_value = patterns['common_corrections'][original_value.lower()]
            return {
                'value': corrected_value,
                'confidence': min(0.95, original_confidence + 0.3),  # Boost confidence
                'method': 'learned_correction'
            }

        # If original extraction failed, try confidence boosters
        if not original_value and patterns['confidence_boosters']:
            # Look for any of the learned patterns in the text
            text_lower = text.lower()
            for booster in patterns['confidence_boosters']:
                if booster.lower() in text_lower:
                    # Extract surrounding context
                    context = self._extract_context(text, booster)
                    if context:
                        return {
                            'value': context,
                            'confidence': 0.75,  # Medium confidence for learned pattern
                            'method': 'learned_pattern'
                        }

        # If still no good result, try improved patterns
        if original_confidence < 0.5 and patterns['improved_patterns']:
            text_lower = text.lower()
            for pattern in patterns['improved_patterns']:
                if pattern.lower() in text_lower:
                    return {
                        'value': pattern,
                        'confidence': 0.70,
                        'method': 'learned_improvement'
                    }

        return field_data

    def _extract_context(self, text, pattern):
        """Extract context around a found pattern"""
        try:
            # Find the pattern in text (case insensitive)
            pattern_lower = pattern.lower()
            text_lower = text.lower()

            start_idx = text_lower.find(pattern_lower)
            if start_idx == -1:
                return None

            # Extract the actual case-preserved text
            end_idx = start_idx + len(pattern)
            return text[start_idx:end_idx].strip()

        except Exception:
            return pattern


def apply_learning_corrections(field_extractor, learning_patterns: Dict):
    """Apply learning patterns to improve field extraction (placeholder for ML)"""
    # This is where you would apply machine learning improvements
    # For now, we'll implement rule-based improvements

    if 'field_accuracy' in learning_patterns:
        for field_name, accuracy in learning_patterns['field_accuracy'].items():
            if accuracy < 0.7:  # If field has low accuracy
                logger.info(f"Field {field_name} needs improvement (accuracy: {accuracy:.2f})")
                # In a real implementation, you would:
                # 1. Retrain ML models with correction data
                # 2. Adjust confidence thresholds
                # 3. Update extraction patterns

    return field_extractor