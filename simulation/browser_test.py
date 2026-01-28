"""브라우저 테스트 - 시뮬레이터 자동 배치 검증"""

import asyncio
from playwright.async_api import async_playwright


async def test_simulation():
    """시뮬레이터 브라우저 테스트"""
    async with async_playwright() as p:
        # 헤드리스 브라우저 실행
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("=" * 50)
        print("시뮬레이터 브라우저 테스트 시작")
        print("=" * 50)

        # 1. 페이지 로드
        print("\n[1] 페이지 로딩...")
        await page.goto("http://localhost:8080/simulation/")
        await page.wait_for_load_state("networkidle")
        print("    ✓ 페이지 로드 완료")

        # 2. 로딩 완료 대기 (PLY 파일 로딩)
        print("\n[2] 가구 데이터 로딩 대기...")
        try:
            # 로딩 화면이 hidden 클래스를 가질 때까지 대기
            await page.wait_for_function(
                "document.querySelector('#loading').classList.contains('hidden')",
                timeout=60000
            )
            print("    ✓ 가구 로딩 완료")
        except Exception as e:
            print(f"    ⚠ 로딩 타임아웃: {e}")

        # 3. 초기 상태 확인
        print("\n[3] 초기 상태 확인...")
        load_percent = await page.text_content("#load-percent")
        placed_count = await page.text_content("#placed-count")
        total_count = await page.text_content("#total-count")
        print(f"    적재율: {load_percent}%")
        print(f"    배치됨: {placed_count}/{total_count}")

        # 4. 스크린샷 (배치 전)
        await page.screenshot(path="/tmp/simulation_before.png")
        print("    ✓ 스크린샷 저장: /tmp/simulation_before.png")

        # 5. 서버 최적화 버튼 클릭
        print("\n[4] 서버 최적화 버튼 클릭...")

        # alert 대화상자 처리
        page.on("dialog", lambda dialog: asyncio.create_task(dialog.accept()))

        await page.click("text=서버 최적화")

        # 로딩 완료 대기
        await asyncio.sleep(2)  # API 호출 대기
        await page.wait_for_function(
            "document.querySelector('#loading').classList.contains('hidden')",
            timeout=30000
        )
        await asyncio.sleep(1)  # UI 업데이트 대기

        # 6. 최적화 후 상태 확인
        print("\n[5] 최적화 후 상태 확인...")
        load_percent_after = await page.text_content("#load-percent")
        placed_count_after = await page.text_content("#placed-count")
        print(f"    적재율: {load_percent_after}%")
        print(f"    배치됨: {placed_count_after}/{total_count}")

        # 7. 스크린샷 (배치 후)
        await page.screenshot(path="/tmp/simulation_after.png")
        print("    ✓ 스크린샷 저장: /tmp/simulation_after.png")

        # 8. 로컬 BLF 테스트
        print("\n[6] 초기화 후 로컬 BLF 테스트...")
        await page.click("text=초기화")
        await asyncio.sleep(0.5)

        page.on("dialog", lambda dialog: asyncio.create_task(dialog.accept()))
        await page.click("text=로컬 BLF")
        await asyncio.sleep(1)

        load_percent_blf = await page.text_content("#load-percent")
        placed_count_blf = await page.text_content("#placed-count")
        print(f"    적재율: {load_percent_blf}%")
        print(f"    배치됨: {placed_count_blf}/{total_count}")

        # 9. 스크린샷 (BLF 후)
        await page.screenshot(path="/tmp/simulation_blf.png")
        print("    ✓ 스크린샷 저장: /tmp/simulation_blf.png")

        # 결과 요약
        print("\n" + "=" * 50)
        print("테스트 결과 요약")
        print("=" * 50)
        print(f"총 가구: {total_count}개")
        print(f"서버 최적화 (py3dbp): {placed_count_after}개 배치, 적재율 {load_percent_after}%")
        print(f"로컬 BLF: {placed_count_blf}개 배치, 적재율 {load_percent_blf}%")
        print("\n스크린샷:")
        print("  - /tmp/simulation_before.png (배치 전)")
        print("  - /tmp/simulation_after.png (서버 최적화 후)")
        print("  - /tmp/simulation_blf.png (로컬 BLF 후)")

        await browser.close()
        print("\n✓ 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_simulation())
