(function () {
  function pinActiveNavOnMobile() {
    if (!window.matchMedia('(max-width: 900px)').matches) return;

    var sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;

    var activeItem = sidebar.querySelector('.nav-item.active');
    if (!activeItem) return;

    var logo = sidebar.querySelector('.logo');
    if (logo && logo.nextSibling !== activeItem) {
      sidebar.insertBefore(activeItem, logo.nextSibling);
      return;
    }

    if (!logo && sidebar.firstChild !== activeItem) {
      sidebar.insertBefore(activeItem, sidebar.firstChild);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', pinActiveNavOnMobile);
  } else {
    pinActiveNavOnMobile();
  }
})();
