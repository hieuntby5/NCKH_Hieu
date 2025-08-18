-- XÓA database cũ nếu tồn tại
DROP DATABASE IF EXISTS qlsinhvien;

-- TẠO database mới
CREATE DATABASE qlsinhvien;
USE qlsinhvien;

-- TẠO bảng lớp với UUID
CREATE TABLE lop (
    id CHAR(36) PRIMARY KEY,
    maLop VARCHAR(10) UNIQUE,
    tenLop VARCHAR(100)
);

-- TẠO bảng sinh viên có FK UUID
CREATE TABLE sinhvien (
    id CHAR(36) PRIMARY KEY,
    maSV VARCHAR(10) UNIQUE,
    hoTen VARCHAR(100),
    ngaySinh DATE,
    gioiTinh ENUM('Nam', 'Nữ'),
    lop_id CHAR(36),
    FOREIGN KEY (lop_id) REFERENCES lop(id)
);

-- Thêm dữ liệu vào bảng lớp (UUID thủ công để dùng lại)
INSERT INTO lop (id, maLop, tenLop) VALUES
('11111111-aaaa-bbbb-cccc-000000000001', 'TĐHCH', 'Tự động hóa chỉ huy'),
('22222222-aaaa-bbbb-cccc-000000000002', 'AI',    'Trí tuệ nhân tạo'),
('33333333-aaaa-bbbb-cccc-000000000003', 'DA',    'Phân tích dữ liệu'),
('44444444-aaaa-bbbb-cccc-000000000004', 'CS',    'An ninh mạng');

-- Thêm dữ liệu vào bảng sinh viên, dùng đúng UUID lớp
INSERT INTO sinhvien (id, maSV, hoTen, ngaySinh, gioiTinh, lop_id) VALUES
(UUID(), 'SV01', 'Hoàng Ngọc Hiếu', '2005-04-07', 'Nam',  '22222222-aaaa-bbbb-cccc-000000000002'),
(UUID(), 'SV02', 'Hảng Thị Mái',    '2005-07-25', 'Nữ',  '22222222-aaaa-bbbb-cccc-000000000002'),
(UUID(), 'SV03', 'Lê Văn Hạnh',     '2005-05-10', 'Nam', '11111111-aaaa-bbbb-cccc-000000000001'),
(UUID(), 'SV04', 'Hoàng Hồng Trâm', '2005-11-19', 'Nữ',  '22222222-aaaa-bbbb-cccc-000000000002'),
(UUID(), 'SV05', 'Hoàng Văn Vũ',    '2005-08-09', 'Nam', '11111111-aaaa-bbbb-cccc-000000000001'),
(UUID(), 'SV06', 'Đặng Thị Sinh',   '2005-03-18', 'Nữ',  '33333333-aaaa-bbbb-cccc-000000000003');

-- XEM toàn bộ dữ liệu bảng lớp
SELECT * FROM lop;

-- XEM toàn bộ dữ liệu bảng sinh viên
SELECT * FROM sinhvien;

-- TRUY VẤN: Lấy danh sách sinh viên của lớp TĐHCH
SELECT sv.maSV, sv.hoTen, l.maLop
FROM sinhvien sv
JOIN lop l ON sv.lop_id = l.id
WHERE l.maLop = 'TĐHCH';
