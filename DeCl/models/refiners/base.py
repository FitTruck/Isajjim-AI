class BaseRefiner:
    def refine(self, bbox, current_subtype):
        """기본은 아무것도 수정하지 않고 그대로 반환"""
        return current_subtype